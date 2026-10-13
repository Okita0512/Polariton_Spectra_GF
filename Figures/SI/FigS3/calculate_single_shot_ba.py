#!/usr/bin/env python3
"""Generate continuous-Brownian single-shot Born spectra at zero temperature."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import numpy as np

from config import (
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
    sigma_map_eV,
)


ROOT = Path(__file__).resolve().parent


def interpolate_complex(
    target: np.ndarray,
    source: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a complex array."""
    return np.interp(target, source, values.real) + 1.0j * np.interp(
        target, source, values.imag
    )


def normalize(values: np.ndarray) -> np.ndarray:
    """Return peak-normalized values for a convenient third data column."""
    maximum = float(np.max(values))
    return values / maximum if maximum > 0.0 else values.copy()


def output_path(lambda_mev: float, gamma_eV: float) -> Path:
    return (
        ROOT
        / "results"
        / (
            f"single_shot_ba_lambda_{lambda_mev:.0f}_"
            f"gamma_{gamma_tag(gamma_eV)}.txt"
        )
    )


def calculate_case(case: tuple[float, float]) -> dict[str, object]:
    """Evaluate the Born kernel once with the bare molecular propagator."""
    lambda_mev, gamma_eV = case
    lambda_eV = lambda_mev * 1.0e-3
    solver_energy = make_continuum_grid()
    sigma_ba = sigma_map_eV(
        solver_energy,
        np.zeros_like(solver_energy, dtype=complex),
        lambda_eV,
        gamma_eV,
    )

    output_energy = np.linspace(
        OUTPUT_OMEGA_MIN_EV,
        OUTPUT_OMEGA_MAX_EV,
        N_OUTPUT_OMEGA,
    )
    output_sigma = interpolate_complex(
        output_energy, solver_energy, sigma_ba
    )
    output_green = electronic_green_eV(
        output_energy, output_sigma, lambda_eV
    )
    output_spectrum = -output_green.imag

    for name, values in (
        ("self-energy", output_sigma),
        ("Green function", output_green),
        ("spectrum", output_spectrum),
    ):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"BA {name} contains NaN or Inf for {case}")
    if float(np.max(output_sigma.imag)) > 1.0e-9:
        raise RuntimeError(f"BA self-energy violates causality for {case}")
    if float(np.min(output_spectrum)) < -1.0e-9:
        raise RuntimeError(f"BA spectrum violates causality for {case}")

    path = output_path(lambda_mev, gamma_eV)
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
            "minus_Im_G_peak_normalized  Re_Sigma_BA_eV  Im_Sigma_BA_eV"
        ),
    )
    area = float(np.trapz(output_spectrum, output_energy))
    print(
        f"Saved {path.name}: area={area:.8f}",
        flush=True,
    )
    return {
        "lambda_meV": f"{lambda_mev:.1f}",
        "Gamma_eV": gamma_tag(gamma_eV),
        "method": "single_shot_BA_continuous_Brownian",
        "spectral_area": f"{area:.16e}",
        "min_spectrum_per_eV": (
            f"{float(np.min(output_spectrum)):.16e}"
        ),
        "max_positive_Im_Sigma_eV": (
            f"{float(np.max(output_sigma.imag)):.16e}"
        ),
    }


def main() -> None:
    (ROOT / "results").mkdir(exist_ok=True)
    cases = [
        (lambda_mev, gamma_eV)
        for gamma_eV in GAMMAS_EV
        for lambda_mev in LAMBDAS_MEV
    ]
    workers = max(
        1, int(os.environ.get("FIGS1_BROWNIAN_BA_WORKERS", "2"))
    )
    if workers == 1:
        rows = [calculate_case(case) for case in cases]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(calculate_case, cases))

    summary = ROOT / "results" / "single_shot_ba_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {summary}")


if __name__ == "__main__":
    main()
