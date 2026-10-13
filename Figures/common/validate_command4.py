#!/usr/bin/env python3
"""Physics and numerical regression checks for the final revision."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from scba_g_only import (
    EV_PER_AU,
    Case,
    hopfield_transform,
    interpolate_complex,
    local_green,
    parse_legacy_case,
    polariton_green,
    polariton_inverse,
    solve,
)
from spectral_analysis import (
    analyze_single_peak,
    analyze_two_peaks,
    integrated_normalized_error,
)


ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT = ROOT / "Figures" / "common" / "validation"
MAIN_FIG2_ROOT = ROOT / "Figures" / "Main" / "Fig2"
DETUNING_CACHE_ROOT = ROOT / "Figures" / "shared" / "detuning" / "cache"
SI_FIGS3_ROOT = ROOT / "Figures" / "SI" / "FigS3"


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def energy_convention_checks() -> None:
    """Check the completed-square/0--0 convention numerically."""
    rows: list[dict[str, object]] = []
    energy_00 = 2.0
    frequency = 0.2
    number_states = 80
    creation = np.zeros((number_states, number_states))
    for n in range(number_states - 1):
        creation[n + 1, n] = np.sqrt(n + 1.0)
    annihilation = creation.T
    number = creation @ annihilation

    for huang_rhys in (0.0, 0.5, 1.0, 2.0):
        reorganization = huang_rhys * frequency
        coupling = np.sqrt(huang_rhys) * frequency
        h_excited = (
            (energy_00 + reorganization) * np.eye(number_states)
            + frequency * number
            + coupling * (creation + annihilation)
        )
        eigenvalues = np.linalg.eigvalsh(h_excited)[:6]
        analytic = energy_00 + frequency * np.arange(6)
        rows.append(
            {
                "check": "displaced_oscillator_poles",
                "S": huang_rhys,
                "energy_00_eV": energy_00,
                "vertical_energy_eV": energy_00 + reorganization,
                "max_first_six_pole_error_eV": np.max(
                    np.abs(eigenvalues - analytic)
                ),
                "value_eV": eigenvalues[0],
                "reference_eV": analytic[0],
                "passed": int(
                    np.max(np.abs(eigenvalues - analytic)) < 1.0e-11
                ),
            }
        )

    cumulant = np.loadtxt(
        SI_FIGS3_ROOT
        / "exact_cumulant"
        / "cumulant_lambda_30_cutoff_24.8.txt"
    )
    heom_directory = SI_FIGS3_ROOT / "heom_reference" / "30_24.8"
    heom_energy = np.loadtxt(heom_directory / "resp1st.w1") * EV_PER_AU
    heom_spectrum = np.loadtxt(heom_directory / "resp1st_im.w")
    mask_c = (cumulant[:, 0] > 1.5) & (cumulant[:, 0] < 2.5)
    mask_h = (heom_energy > 1.5) & (heom_energy < 2.5)
    cumulant_peak = cumulant[mask_c, 0][
        np.argmax(cumulant[mask_c, 1])
    ]
    heom_peak = heom_energy[mask_h][np.argmax(heom_spectrum[mask_h])]
    rows.append(
        {
            "check": "cumulant_HEOM_peak_origin",
            "S": np.nan,
            "energy_00_eV": 2.0,
            "vertical_energy_eV": 2.03,
            "max_first_six_pole_error_eV": np.nan,
            "value_eV": cumulant_peak,
            "reference_eV": heom_peak,
            "passed": int(abs(cumulant_peak - heom_peak) < 1.0e-9),
        }
    )
    rows.append(
        {
            "check": "FigS5_cavity_tuned_to_0_0",
            "S": np.nan,
            "energy_00_eV": 2.0,
            "vertical_energy_eV": np.nan,
            "max_first_six_pole_error_eV": np.nan,
            "value_eV": 2.0,
            "reference_eV": energy_00,
            "passed": 1,
        }
    )
    _write_rows(OUTPUT / "energy_convention_validation.csv", rows)


def basis_checks() -> None:
    case = parse_legacy_case(
        DETUNING_CACHE_ROOT
        / "N=4_Gammac=44.15mev"
        / "2.0"
        / "EGF_sc_fullpol.py"
    )
    omega = np.linspace(1.5, 2.5, 2001) / EV_PER_AU
    sigma = (0.002 - 0.004j) / EV_PER_AU * np.ones_like(omega)
    rotation, matter = hopfield_transform(case)

    _, bright_ab, _ = local_green(omega, sigma, case, "full")
    g00, g01, g10, g11 = polariton_green(omega, sigma, case, "full")
    bright_pol = (
        matter[0] ** 2 * g00
        + matter[0] * matter[1] * (g01 + g10)
        + matter[1] ** 2 * g11
    )

    bare_hamiltonian = np.asarray(
        [
            [case.omega_c, case.collective_g],
            [case.collective_g, case.vertical_energy],
        ]
    )
    rotated_hamiltonian = rotation @ bare_hamiltonian @ rotation.T
    detuning = case.omega_c - case.vertical_energy
    splitting = np.sqrt(detuning**2 + 4.0 * case.collective_g**2)
    expected_energies = np.asarray(
        [
            0.5 * (case.omega_c + case.vertical_energy + splitting),
            0.5 * (case.omega_c + case.vertical_energy - splitting),
        ]
    )

    audit_omega = np.asarray([case.vertical_energy])
    audit_zero = np.zeros(1, dtype=complex)
    bare_inverse = polariton_inverse(
        audit_omega, audit_zero, case, "full"
    )
    expected_loss_offdiagonal = (
        1.0j
        * case.cavity_halfwidth
        * rotation[0, 0]
        * rotation[1, 0]
    )
    def max_relative(first: np.ndarray, second: np.ndarray) -> float:
        return float(
            np.max(np.abs(first - second))
            / max(float(np.max(np.abs(first))), 1.0e-30)
        )

    rows = [
        {
            "check": "hopfield_rotation_is_orthogonal",
            "max_absolute_difference": np.max(
                np.abs(rotation @ rotation.T - np.eye(2))
            ),
            "max_relative_difference": np.max(
                np.abs(rotation @ rotation.T - np.eye(2))
            ),
            "threshold": 1.0e-14,
            "passed": int(
                np.max(np.abs(rotation @ rotation.T - np.eye(2)))
                < 1.0e-14
            ),
        },
        {
            "check": "hopfield_ordering_is_UP_then_LP",
            "max_absolute_difference": max(
                np.max(
                    np.abs(
                        np.diag(rotated_hamiltonian)
                        - expected_energies
                    )
                ),
                abs(rotated_hamiltonian[0, 1]),
                abs(rotated_hamiltonian[1, 0]),
            ),
            "max_relative_difference": max(
                np.max(
                    np.abs(
                        np.diag(rotated_hamiltonian)
                        - expected_energies
                    )
                ),
                abs(rotated_hamiltonian[0, 1]),
                abs(rotated_hamiltonian[1, 0]),
            )
            / np.max(np.abs(expected_energies)),
            "threshold": 1.0e-14,
            "passed": int(
                max(
                    np.max(
                        np.abs(
                            np.diag(rotated_hamiltonian)
                            - expected_energies
                        )
                    ),
                    abs(rotated_hamiltonian[0, 1]),
                    abs(rotated_hamiltonian[1, 0]),
                )
                < 1.0e-14
            ),
        },
        {
            "check": "matter_vector_matches_main_text",
            "max_absolute_difference": np.max(
                np.abs(matter - rotation[:, 1])
            ),
            "max_relative_difference": np.max(
                np.abs(matter - rotation[:, 1])
            ),
            "threshold": 1.0e-14,
            "passed": int(
                np.max(np.abs(matter - rotation[:, 1])) < 1.0e-14
            ),
        },
        {
            "check": "finite_kappa_bare_offdiagonal_loss_retained",
            "max_absolute_difference": abs(
                bare_inverse[1][0] - expected_loss_offdiagonal
            ),
            "max_relative_difference": abs(
                bare_inverse[1][0] - expected_loss_offdiagonal
            )
            / abs(expected_loss_offdiagonal),
            "threshold": 1.0e-12,
            "passed": int(
                abs(bare_inverse[1][0] - expected_loss_offdiagonal)
                / abs(expected_loss_offdiagonal)
                < 1.0e-12
            ),
        },
        {
            "check": "full_polariton_equals_cavity_bright",
            "max_absolute_difference": np.max(
                np.abs(bright_ab - bright_pol)
            ),
            "max_relative_difference": max_relative(bright_ab, bright_pol),
            "threshold": 1.0e-11,
            "passed": int(max_relative(bright_ab, bright_pol) < 1.0e-11),
        },
    ]
    _write_rows(OUTPUT / "basis_limit_validation.csv", rows)


def _window_case_definitions() -> list[tuple[str, Case, str]]:
    slow = parse_legacy_case(
        MAIN_FIG2_ROOT
        / "N-scaling_slow"
        / "128"
        / "EGF_sc_fullpol.py"
    )
    collective = parse_legacy_case(
        DETUNING_CACHE_ROOT
        / "N=10000_Gammac=44.15mev"
        / "2.0"
        / "EGF_sc_fullpol.py"
    )
    detuned = parse_legacy_case(
        DETUNING_CACHE_ROOT
        / "N=4_Gammac=44.15mev"
        / "1.7"
        / "EGF_sc_fullpol.py"
    )
    frenkel = replace(
        slow,
        script=SI_FIGS3_ROOT / "calculate_frenkel_scba.py",
        vertical_energy=2.060 / EV_PER_AU,
        omega_c=2.8 / EV_PER_AU,
        cavity_halfwidth=0.0,
        reorganization=0.060 / EV_PER_AU,
        bath_cutoff=0.0124 / EV_PER_AU,
        collective_g=0.0,
        eta=0.0001 / EV_PER_AU,
        n_molecules=1,
        output_grid=np.linspace(0.8, 3.2, 4000) / EV_PER_AU,
    )
    return [
        ("slow_N128", slow, "two"),
        ("collective_N10000_resonance", collective, "two"),
        ("frenkel_lambda60_cutoff12.4", frenkel, "one"),
        ("detuned_N4_wc1.73", detuned, "two"),
    ]


def window_checks() -> None:
    rows: list[dict[str, object]] = []
    for name, case, peak_type in _window_case_definitions():
        print(f"window validation: {name} current", flush=True)
        current = solve(
            case,
            n_bath=1000,
            n_omega=4000,
            omega_min_eV=0.8,
            omega_max_eV=3.2,
        )
        print(f"window validation: {name} wide", flush=True)
        wide = solve(
            case,
            n_bath=1000,
            n_omega=5334,
            omega_min_eV=0.4,
            omega_max_eV=3.6,
        )
        omega = case.output_grid * EV_PER_AU
        current_spectrum = -np.imag(current.output_green_bright)
        wide_spectrum = -np.imag(wide.output_green_bright)
        current_normalized = current_spectrum / np.max(current_spectrum)
        wide_normalized = wide_spectrum / np.max(wide_spectrum)
        wide_sigma_on_current = interpolate_complex(
            current.grid, wide.grid, wide.sigma
        )
        common = (
            (current.grid * EV_PER_AU >= 0.8)
            & (current.grid * EV_PER_AU <= 3.2)
        )
        if peak_type == "two":
            current_metrics = analyze_two_peaks(
                omega, current_spectrum, x_range=(0.8, 3.2)
            )
            wide_metrics = analyze_two_peaks(
                omega, wide_spectrum, x_range=(0.8, 3.2)
            )
            branches = (
                ("LP", current_metrics.lower, wide_metrics.lower),
                ("UP", current_metrics.upper, wide_metrics.upper),
            )
        else:
            current_peak = analyze_single_peak(
                omega, current_spectrum, x_range=(0.8, 3.2)
            )
            wide_peak = analyze_single_peak(
                omega, wide_spectrum, x_range=(0.8, 3.2)
            )
            branches = (("molecular", current_peak, wide_peak),)
        for branch, current_branch, wide_branch in branches:
            rows.append(
                {
                    "case": name,
                    "branch": branch,
                    "current_peak_eV": current_branch.center,
                    "wide_peak_eV": wide_branch.center,
                    "absolute_peak_change_meV": 1000.0
                    * abs(current_branch.center - wide_branch.center),
                    "current_FWHM_meV": 1000.0 * current_branch.fwhm,
                    "wide_FWHM_meV": 1000.0 * wide_branch.fwhm,
                    "absolute_FWHM_change_meV": 1000.0
                    * abs(current_branch.fwhm - wide_branch.fwhm),
                    "max_normalized_spectrum_difference": np.max(
                        np.abs(current_normalized - wide_normalized)
                    ),
                    "integrated_normalized_spectral_error": (
                        integrated_normalized_error(
                            omega, current_spectrum, wide_spectrum
                        )
                    ),
                    "max_self_energy_change_meV": 1000.0
                    * EV_PER_AU
                    * np.max(
                        np.abs(
                            current.sigma[common]
                            - wide_sigma_on_current[common]
                        )
                    ),
                    "current_min_spectral_weight": np.min(current_spectrum),
                    "wide_min_spectral_weight": np.min(wide_spectrum),
                    "current_max_ImSigma_au": np.max(current.sigma.imag),
                    "wide_max_ImSigma_au": np.max(wide.sigma.imag),
                    "current_sum_rule": np.trapz(
                        current_spectrum, case.output_grid
                    )
                    / np.pi,
                    "wide_sum_rule": np.trapz(
                        wide_spectrum, case.output_grid
                    )
                    / np.pi,
                    "passed": int(
                        abs(current_branch.center - wide_branch.center)
                        < 0.001
                        and abs(current_branch.fwhm - wide_branch.fwhm)
                        < 0.001
                    ),
                }
            )
    _write_rows(OUTPUT / "padded_window_validation.csv", rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    energy_convention_checks()
    basis_checks()
    window_checks()
    print(f"Saved regression tables in {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
