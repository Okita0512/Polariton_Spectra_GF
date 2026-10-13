#!/usr/bin/env python3
"""Grid and alternative-quadrature checks for production Drude SCBA."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scba_g_only import EV_PER_AU, parse_legacy_case, solve
from spectral_analysis import analyze_single_peak, analyze_two_peaks


ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT = ROOT / "Figures" / "common" / "validation"
MAIN_FIG2_ROOT = ROOT / "Figures" / "Main" / "Fig2"
DETUNING_CACHE_ROOT = ROOT / "Figures" / "shared" / "detuning" / "cache"
SI_FIGS12_ROOT = ROOT / "Figures" / "SI" / "FigS12"


CASES = (
    (
        "SI Fig. S12, N=1",
        SI_FIGS12_ROOT / "N=1" / "EGF_sc_fullpol.py",
        "two",
    ),
    (
        "main Fig. 2 slow bath, N=128",
        MAIN_FIG2_ROOT
        / "N-scaling_slow"
        / "128"
        / "EGF_sc_fullpol.py",
        "two",
    ),
    (
        "detuning cache, N=64, wc=2.0 eV",
        DETUNING_CACHE_ROOT
        / "N=64_Gammac=44.15mev"
        / "2.0"
        / "EGF_sc_fullpol.py",
        "two",
    ),
    (
        "main Fig. 2 bare fast-bath molecule",
        MAIN_FIG2_ROOT
        / "fast"
        / "N=1_out"
        / "EGF_sc_fullpol.py",
        "one",
    ),
)


def normalized(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    area = float(np.trapz(values, x))
    if area <= 0.0:
        raise RuntimeError("nonpositive spectral area")
    return values / area


def spectrum_metrics(
    energy: np.ndarray,
    spectrum: np.ndarray,
    branch_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    if branch_mode == "one":
        metric = analyze_single_peak(energy, spectrum)
        return (
            np.asarray([metric.center]),
            np.asarray([metric.fwhm]),
        )
    metrics = analyze_two_peaks(energy, spectrum)
    return (
        np.asarray([metrics.lower.center, metrics.upper.center]),
        np.asarray([metrics.lower.fwhm, metrics.upper.fwhm]),
    )


def comparison(
    label: str,
    reference,
    candidate,
    case,
    branch_mode: str,
    comparison_name: str,
) -> dict[str, object]:
    energy = case.output_grid * EV_PER_AU
    reference_spectrum = -np.imag(reference.output_green_bright)
    candidate_spectrum = -np.imag(candidate.output_green_bright)
    reference_norm = normalized(reference_spectrum, energy)
    candidate_norm = normalized(candidate_spectrum, energy)
    reference_peaks, reference_widths = spectrum_metrics(
        energy, reference_spectrum, branch_mode
    )
    candidate_peaks, candidate_widths = spectrum_metrics(
        energy, candidate_spectrum, branch_mode
    )
    return {
        "case": label,
        "comparison": comparison_name,
        "production_step_meV": (
            (reference.grid[1] - reference.grid[0])
            * EV_PER_AU
            * 1.0e3
        ),
        "candidate_step_meV": (
            (candidate.grid[1] - candidate.grid[0])
            * EV_PER_AU
            * 1.0e3
        ),
        "max_peak_change_meV": (
            np.max(np.abs(candidate_peaks - reference_peaks)) * 1.0e3
        ),
        "max_fwhm_change_meV": (
            np.max(np.abs(candidate_widths - reference_widths)) * 1.0e3
        ),
        "area_normalized_L1_change": float(
            np.trapz(np.abs(candidate_norm - reference_norm), energy)
        ),
        "max_area_normalized_pointwise_change": float(
            np.max(np.abs(candidate_norm - reference_norm))
        ),
        "max_self_energy_change_meV": float(
            np.max(
                np.abs(
                    candidate.output_sigma - reference.output_sigma
                )
            )
            * EV_PER_AU
            * 1.0e3
        ),
        "production_iterations": reference.iterations,
        "candidate_iterations": candidate.iterations,
        "production_residual_meV": (
            reference.final_error * EV_PER_AU * 1.0e3
        ),
        "candidate_residual_meV": (
            candidate.final_error * EV_PER_AU * 1.0e3
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for label, script, branch_mode in CASES:
        case = parse_legacy_case(script)
        production_step = min(
            0.05, case.eta * EV_PER_AU * 1.0e3
        )
        print(f"{label}: production convolution", flush=True)
        production = solve(
            case,
            bath_method="convolution",
            omega_step_meV=production_step,
        )
        print(f"{label}: half-step convolution", flush=True)
        refined = solve(
            case,
            bath_method="convolution",
            omega_step_meV=0.5 * production_step,
        )
        rows.append(
            comparison(
                label,
                production,
                refined,
                case,
                branch_mode,
                "half_step_convolution",
            )
        )

        print(f"{label}: tangent alternative", flush=True)
        tangent = solve(
            case,
            bath_method="tangent",
            n_bath=2000,
            n_omega=8000,
            omega_min_eV=-1.0,
            omega_max_eV=4.0,
        )
        rows.append(
            comparison(
                label,
                production,
                tangent,
                case,
                branch_mode,
                "tangent_Nb2000_Nw8000",
            )
        )

    path = OUTPUT / "scba_convolution_validation.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
