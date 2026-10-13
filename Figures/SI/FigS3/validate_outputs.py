#!/usr/bin/env python3
"""Validate completeness, conventions, sum rules, and audit status."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from brownian import (
    exact_brownian_lineshape,
    spectral_density_eV,
    spectral_density_over_frequency_au,
)
from config import (
    CENTER_FREQUENCY_EV,
    CONTINUUM_OMEGA_MAX_EV,
    CONTINUUM_OMEGA_MIN_EV,
    CONTINUUM_STEP_EV,
    EV_PER_AU,
    GAMMAS_EV,
    LAMBDAS_MEV,
    N_OUTPUT_OMEGA,
    OUTPUT_OMEGA_MAX_EV,
    OUTPUT_OMEGA_MIN_EV,
    gamma_tag,
)


def independent_cumulant_quadrature_error() -> float:
    """Compare the FFT/analytic cumulant with direct adaptive quadrature."""
    lambda_mev = 400.0
    gamma_eV = 0.1
    lam = lambda_mev * 1.0e-3 / EV_PER_AU
    gamma = gamma_eV / EV_PER_AU
    center = CENTER_FREQUENCY_EV / EV_PER_AU
    # Leave a factor-ten time margin after the latest comparison point so
    # the auxiliary transform cannot wrap into the validation interval.
    time = np.arange(1002, dtype=float)
    transformed = exact_brownian_lineshape(
        time,
        lambda_mev,
        gamma_eV,
        fft_multiplier=16,
        substeps=8,
    )
    errors = []
    for index in (1, 10, 100):
        scaled_time = center * time[index]

        def h_scaled(x: float) -> float:
            return float(
                spectral_density_over_frequency_au(
                    np.array([center * x]),
                    lam,
                    gamma,
                    center,
                )[0]
            )

        real = quad(
            lambda x: (
                h_scaled(x)
                * 2.0
                * np.sin(0.5 * scaled_time * x) ** 2
                / x
            ),
            0.0,
            2000.0,
            epsabs=2.0e-10,
            epsrel=2.0e-10,
            limit=3000,
        )[0] / np.pi
        imag = quad(
            lambda x: (
                h_scaled(x)
                * (
                    np.sin(scaled_time * x)
                    - scaled_time * x
                )
                / x
            ),
            0.0,
            2000.0,
            epsabs=2.0e-10,
            epsrel=2.0e-10,
            limit=3000,
        )[0] / np.pi
        errors.append(
            abs(transformed[index] - (real + 1.0j * imag))
        )
    return float(max(errors))


def main() -> None:
    root = Path(__file__).resolve().parent
    cases = {
        (lambda_mev, gamma_eV)
        for gamma_eV in GAMMAS_EV
        for lambda_mev in LAMBDAS_MEV
    }
    exact_areas = []
    ba_areas = []
    scba_areas = []
    density_errors = []
    for lambda_mev, gamma_eV in sorted(cases):
        lambda_eV = lambda_mev * 1e-3
        numerical_lambda = quad(
            lambda w: float(
                spectral_density_eV(
                    np.array([w]), lambda_eV, gamma_eV
                )[0]
                / w
                / np.pi
            ),
            0.0,
            np.inf,
            epsabs=1.0e-12,
            epsrel=1.0e-12,
            limit=500,
        )[0]
        density_errors.append(abs(numerical_lambda - lambda_eV))

        exact = np.loadtxt(
            root
            / "exact_cumulant"
            / (
                f"cumulant_lambda_{lambda_mev:.0f}_"
                f"gamma_{gamma_tag(gamma_eV)}.txt"
            )
        )
        scba = np.loadtxt(
            root
            / "results"
            / (
                f"scba_lambda_{lambda_mev:.0f}_"
                f"gamma_{gamma_tag(gamma_eV)}.txt"
            )
        )
        ba = np.loadtxt(
            root
            / "results"
            / (
                f"single_shot_ba_lambda_{lambda_mev:.0f}_"
                f"gamma_{gamma_tag(gamma_eV)}.txt"
            )
        )
        if (
            not np.all(np.isfinite(exact))
            or not np.all(np.isfinite(ba))
            or not np.all(np.isfinite(scba))
        ):
            raise RuntimeError(f"non-finite spectrum for {(lambda_mev, gamma_eV)}")
        if float(np.min(exact[:, 1])) < -1.0e-7:
            raise RuntimeError(f"negative exact spectrum for {(lambda_mev, gamma_eV)}")
        if float(np.min(ba[:, 1])) < -1.0e-9:
            raise RuntimeError(f"negative BA spectrum for {(lambda_mev, gamma_eV)}")
        if float(np.min(scba[:, 1])) < -1.0e-9:
            raise RuntimeError(f"negative SCBA spectrum for {(lambda_mev, gamma_eV)}")
        if not np.isclose(np.max(exact[:, 2]), 1.0, atol=1.0e-10):
            raise RuntimeError("exact peak normalization failed")
        if not np.isclose(np.max(ba[:, 2]), 1.0, atol=1.0e-10):
            raise RuntimeError("BA peak normalization failed")
        if not np.isclose(np.max(scba[:, 2]), 1.0, atol=1.0e-10):
            raise RuntimeError("SCBA peak normalization failed")
        for method, data in (("BA", ba), ("SCBA", scba)):
            if len(data) != N_OUTPUT_OMEGA:
                raise RuntimeError(f"{method} output grid has the wrong length")
            if (
                not np.isclose(data[0, 0], OUTPUT_OMEGA_MIN_EV)
                or not np.isclose(data[-1, 0], OUTPUT_OMEGA_MAX_EV)
            ):
                raise RuntimeError(f"{method} output grid has the wrong range")
        exact_areas.append(float(np.trapz(exact[:, 1], exact[:, 0])))
        ba_areas.append(float(np.trapz(ba[:, 1], ba[:, 0])))
        scba_areas.append(float(np.trapz(scba[:, 1], scba[:, 0])))

    if max(density_errors) > 1.0e-10:
        raise RuntimeError("Brownian reorganization convention failed")
    cumulant_quadrature_error = independent_cumulant_quadrature_error()
    if cumulant_quadrature_error > 1.0e-6:
        raise RuntimeError("exact cumulant failed direct quadrature")
    summary_path = root / "results" / "scba_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(cases):
        raise RuntimeError("SCBA summary is incomplete")
    if any(str(row["converged"]).lower() != "true" for row in rows):
        raise RuntimeError("SCBA summary includes an unconverged case")
    if any(
        row["bath_treatment"] != "continuous_Brownian_FFT_Volterra"
        for row in rows
    ):
        raise RuntimeError("SCBA bath treatment is stale")
    ba_summary_path = root / "results" / "single_shot_ba_summary.csv"
    with ba_summary_path.open(newline="", encoding="utf-8") as handle:
        ba_rows = list(csv.DictReader(handle))
    if len(ba_rows) != len(cases):
        raise RuntimeError("BA summary is incomplete")
    if any(
        not np.isclose(
            float(row["solver_omega_min_eV"]),
            CONTINUUM_OMEGA_MIN_EV,
        )
        or not np.isclose(
            float(row["solver_omega_max_eV"]),
            CONTINUUM_OMEGA_MAX_EV,
        )
        or not np.isclose(
            float(row["solver_step_meV"]),
            CONTINUUM_STEP_EV * 1e3,
        )
        for row in rows
    ):
        raise RuntimeError("SCBA solver grid is stale")
    convergence_text = (
        root / "convergence" / "convergence_report.txt"
    ).read_text(encoding="utf-8")
    if "status = PASS" not in convergence_text:
        raise RuntimeError("convergence audit did not pass")

    max_exact_sum_error = max(abs(area - np.pi) for area in exact_areas)
    max_ba_sum_error = max(abs(area - np.pi) for area in ba_areas)
    max_scba_sum_error = max(abs(area - np.pi) for area in scba_areas)
    if max_exact_sum_error > 0.02:
        raise RuntimeError("exact spectral sum rule failed")
    if max_scba_sum_error > 0.10:
        raise RuntimeError("SCBA spectral sum rule failed")
    if max_ba_sum_error > 0.10:
        raise RuntimeError("BA spectral sum rule failed")

    report = root / "validation.txt"
    report.write_text(
        "\n".join(
            (
                "Fig. S4 Brownian standalone validation",
                f"case_count_per_method = {len(cases)}",
                "methods = exact_cumulant, single_shot_BA, SCBA",
                "temperature_K = 0",
                f"omega_s_eV = {CENTER_FREQUENCY_EV:.1f}",
                (
                    "max_reorganization_sum_rule_error_eV = "
                    f"{max(density_errors):.16e}"
                ),
                (
                    "max_direct_cumulant_quadrature_error = "
                    f"{cumulant_quadrature_error:.16e}"
                ),
                (
                    "max_exact_spectral_sum_rule_error = "
                    f"{max_exact_sum_error:.16e}"
                ),
                (
                    "max_BA_spectral_sum_rule_error = "
                    f"{max_ba_sum_error:.16e}"
                ),
                (
                    "max_SCBA_spectral_sum_rule_error = "
                    f"{max_scba_sum_error:.16e}"
                ),
                "SCBA_production_bath = continuous_Brownian_integral",
                "discrete_bath_modes = not_used",
                "independent_convergence_audit = PASS",
                "status = PASS",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(report.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
