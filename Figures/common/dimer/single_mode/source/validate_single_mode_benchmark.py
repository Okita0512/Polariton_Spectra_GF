#!/usr/bin/env python3
"""Numerical validation suite for the single-mode Frenkel-dimer benchmark.

The quick profile is intended for development and CI-like checks.  The
default profile uses more ED truncations, a finer SCBA reference grid, and a
longer signed-J continuation scan.  Both profiles exercise the same physical
and numerical invariants and write the same machine-readable outputs.

All paths are resolved from this file.  The historical FigS5 data are read
only; every generated file is confined to
``Figures/common/dimer/single_mode/source``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import ive


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import calculate_single_mode_ed as ed  # noqa: E402
import calculate_single_mode_scba as scba  # noqa: E402
from single_mode_common import (  # noqa: E402
    GridSpec,
    ModelParams,
    RESULTS,
    ROOT,
    VALIDATION,
    default_grid_spec,
    ensure_directories,
    recommended_n_vib,
    thermal_cutoff,
)
from single_mode_metrics import compare_spectra  # noqa: E402


PROJECT_ROOT = ROOT.parents[4]
ARCHIVED_FIGS5 = PROJECT_ROOT / "bkup" / "Unused Figures" / "FigS5"
ANALYTIC_JSON = VALIDATION / "analytic_validation.json"
ED_CONVERGENCE_CSV = RESULTS / "ed_convergence_summary.csv"
SCBA_CONVERGENCE_CSV = RESULTS / "scba_convergence_summary.csv"
FIGS5_REGRESSION_CSV = VALIDATION / "figs5_regression.csv"
NUMERICAL_REPORT = VALIDATION / "numerical_validation.txt"


@dataclass(frozen=True)
class Check:
    """One explicit pass/fail criterion written to the text audit."""

    name: str
    value: float
    comparison: str
    threshold: float
    passed: bool
    detail: str = ""


def _less_equal_check(
    checks: List[Check],
    name: str,
    value: float,
    threshold: float,
    detail: str = "",
) -> bool:
    passed = bool(np.isfinite(value) and value <= threshold)
    checks.append(Check(name, float(value), "<=", float(threshold), passed, detail))
    return passed


def _greater_equal_check(
    checks: List[Check],
    name: str,
    value: float,
    threshold: float,
    detail: str = "",
) -> bool:
    passed = bool(np.isfinite(value) and value >= threshold)
    checks.append(Check(name, float(value), ">=", float(threshold), passed, detail))
    return passed


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, values: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(values), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty validation CSV")
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _relative_complex_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(reference))))
    return float(np.max(np.abs(candidate - reference)) / scale)


def _dominant_peak_energy(
    grid: np.ndarray,
    spectrum: np.ndarray,
    window: Tuple[float, float],
) -> float:
    mask = (grid >= window[0]) & (grid <= window[1])
    if not np.any(mask):
        raise ValueError("Peak window does not intersect the grid")
    indices = np.flatnonzero(mask)
    return float(grid[indices[int(np.argmax(spectrum[mask]))]])


def _compare_on_reference_grid(
    reference_grid: np.ndarray,
    reference_spectrum: np.ndarray,
    candidate_grid: np.ndarray,
    candidate_spectrum: np.ndarray,
    window: Tuple[float, float],
) -> Dict[str, float]:
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(window[0]), abs(window[1]))
    if candidate_grid[0] > window[0] + tolerance or candidate_grid[-1] < window[1] - tolerance:
        raise ValueError("Candidate grid does not cover the comparison window")
    candidate_on_reference = np.interp(
        reference_grid, candidate_grid, candidate_spectrum
    )
    comparison = compare_spectra(
        reference_grid,
        reference_spectrum,
        candidate_on_reference,
        window,
    )
    reference_peak = _dominant_peak_energy(
        reference_grid, reference_spectrum, window
    )
    candidate_peak = _dominant_peak_energy(
        reference_grid, candidate_on_reference, window
    )
    return {
        "reference_raw_area": comparison.reference_raw_area,
        "candidate_raw_area": comparison.candidate_raw_area,
        "raw_area_relative_difference": comparison.raw_area_relative_difference,
        "unit_area_l1": comparison.unit_area_l1,
        "unit_area_l2_per_sqrt_eV": comparison.unit_area_l2_per_sqrt_eV,
        "bhattacharyya_coefficient": comparison.bhattacharyya_coefficient,
        "wasserstein_meV": comparison.wasserstein_eV * 1.0e3,
        "centroid_error_meV": comparison.centroid_error_eV * 1.0e3,
        "dominant_peak_shift_meV": (candidate_peak - reference_peak) * 1.0e3,
    }


def _thermal_bessel_measure(
    model: ModelParams,
    tolerance: float = 1.0e-14,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    r"""Return the exact finite-T independent-boson hot-band progression.

    For integer sideband index l,

      w_l = exp[-S(2n+1)] I_|l|(2S sqrt(n(n+1)))
            [(n+1)/n]^(l/2).

    The ratio w_-l/w_l = exp(-beta*l*omega0) explicitly checks detailed
    balance.  ``ive`` is used so the modified-Bessel exponential is handled
    in log space.
    """

    occupation = model.bose_occupation
    if occupation <= 0.0:
        raise ValueError("The Bessel hot-band check requires finite temperature")
    argument = 2.0 * model.huang_rhys * np.sqrt(
        occupation * (occupation + 1.0)
    )
    log_ratio = np.log((occupation + 1.0) / occupation)
    maximum_sideband = 8
    while True:
        indices = np.arange(-maximum_sideband, maximum_sideband + 1, dtype=int)
        scaled_bessel = ive(np.abs(indices), argument)
        weights = np.zeros(indices.shape, dtype=float)
        positive = scaled_bessel > 0.0
        log_weights = (
            -model.huang_rhys * (2.0 * occupation + 1.0)
            + abs(argument)
            + np.log(scaled_bessel[positive])
            + 0.5 * indices[positive].astype(float) * log_ratio
        )
        weights[positive] = np.exp(log_weights)
        retained = float(np.sum(weights))
        edge = float(max(weights[0], weights[-1]))
        if 1.0 - retained < tolerance and edge < tolerance:
            break
        maximum_sideband *= 2
        if maximum_sideband > 2048:
            raise RuntimeError("Thermal Bessel progression failed to converge")

    detail_balance_error = 0.0
    beta_omega = log_ratio
    for sideband in range(1, maximum_sideband + 1):
        positive_weight = weights[maximum_sideband + sideband]
        negative_weight = weights[maximum_sideband - sideband]
        if positive_weight > 1.0e-300:
            expected_ratio = np.exp(-beta_omega * sideband)
            detail_balance_error = max(
                detail_balance_error,
                abs(negative_weight / positive_weight - expected_ratio),
            )
    poles = model.energy_00_eV + indices.astype(float) * model.omega0_eV
    return poles, weights, {
        "maximum_abs_sideband": float(maximum_sideband),
        "retained_weight": retained,
        "omitted_weight_bound": float(max(0.0, 1.0 - retained)),
        "detailed_balance_max_abs_error": float(detail_balance_error),
    }


def run_finite_temperature_fc_validation(
    quick: bool,
    checks: List[Check],
) -> Dict[str, Any]:
    model = ModelParams(
        energy_00_eV=2.0,
        omega0_meV=24.8,
        huang_rhys=1.0,
        j_meV=0.0,
        temperature_K=300.0,
        eta_meV=1.0,
    )
    grid_spec = default_grid_spec(
        model, step_meV=0.20 if quick else 0.10, padding_eV=0.75
    )
    n_thermal, thermal_omitted = thermal_cutoff(model, 1.0e-10)
    n_vib = recommended_n_vib(model, n_thermal) + (2 if quick else 6)
    result = ed.solve_ed(
        model,
        grid_spec,
        n_vib=n_vib,
        n_thermal=n_thermal,
        deposition_refinement=2 if quick else 4,
    )
    poles, weights, bessel = _thermal_bessel_measure(model)
    analytic_green = ed.direct_lorentz_green(
        result.energy_eV, poles, weights, model.eta_eV
    )
    window = (grid_spec.analysis_min_eV, grid_spec.analysis_max_eV)
    comparison = _compare_on_reference_grid(
        result.energy_eV,
        -np.imag(analytic_green),
        result.energy_eV,
        result.spectrum_bright,
        window,
    )
    l1_threshold = 3.0e-3 if quick else 1.5e-3
    wasserstein_threshold_meV = 0.10 if quick else 0.05
    _less_equal_check(
        checks,
        "finite_T_J0_Bessel_unit_area_L1",
        comparison["unit_area_l1"],
        l1_threshold,
    )
    _less_equal_check(
        checks,
        "finite_T_J0_Bessel_Wasserstein_meV",
        comparison["wasserstein_meV"],
        wasserstein_threshold_meV,
    )
    _less_equal_check(
        checks,
        "finite_T_J0_thermal_omission",
        thermal_omitted,
        1.0e-10,
    )
    _less_equal_check(
        checks,
        "finite_T_J0_Bessel_weight_omission",
        bessel["omitted_weight_bound"],
        2.0e-14,
    )
    _less_equal_check(
        checks,
        "finite_T_J0_Bessel_detailed_balance",
        bessel["detailed_balance_max_abs_error"],
        5.0e-13,
    )
    return {
        "model": asdict(model),
        "grid": asdict(grid_spec),
        "n_vib": n_vib,
        "n_thermal": n_thermal,
        "thermal_omitted_probability": thermal_omitted,
        "bessel_progression": bessel,
        "comparison": comparison,
    }


def run_analytic_validations(
    quick: bool,
    checks: List[Check],
) -> Dict[str, Any]:
    ed_suite = ed.run_validation_suite()
    for name, values in ed_suite.items():
        finite_values = [
            float(value)
            for key, value in values.items()
            if key not in ("maximum_total_quanta", "phonon_dimension", "matrix_dimension", "analytic_sidebands")
            and isinstance(value, (int, float, np.integer, np.floating))
        ]
        all_finite = float(all(np.isfinite(value) for value in finite_values))
        _greater_equal_check(checks, "ED_suite_%s_finite" % name, all_finite, 1.0)

    s_zero = ModelParams(
        omega0_meV=24.8,
        huang_rhys=0.0,
        j_meV=25.0,
        temperature_K=300.0,
        eta_meV=1.0,
    )
    s_zero_grid = default_grid_spec(s_zero, step_meV=0.20, padding_eV=0.55)
    controls = scba.ScbaNumerics(max_iterations=2000)
    ba_zero = scba.solve_ba(s_zero, s_zero_grid, controls)
    scba_zero = scba.solve_scba(s_zero, s_zero_grid, controls)
    analytic_bright = 1.0 / (
        scba_zero.omega_eV
        - (s_zero.vertical_energy_eV + s_zero.j_eV)
        + 1.0j * s_zero.eta_eV
    )
    analytic_dark = 1.0 / (
        scba_zero.omega_eV
        - (s_zero.vertical_energy_eV - s_zero.j_eV)
        + 1.0j * s_zero.eta_eV
    )
    s_zero_errors = {
        "ba_bright_relative_error": _relative_complex_error(
            ba_zero.green_bright, analytic_bright
        ),
        "ba_dark_relative_error": _relative_complex_error(
            ba_zero.green_dark, analytic_dark
        ),
        "scba_bright_relative_error": _relative_complex_error(
            scba_zero.green_bright, analytic_bright
        ),
        "scba_dark_relative_error": _relative_complex_error(
            scba_zero.green_dark, analytic_dark
        ),
        "max_abs_ba_sigma_eV": float(np.max(np.abs(ba_zero.sigma_eV))),
        "max_abs_scba_sigma_eV": float(np.max(np.abs(scba_zero.sigma_eV))),
    }
    _less_equal_check(
        checks,
        "SCBA_S0_analytic_max_relative_error",
        max(s_zero_errors.values()),
        2.0e-13,
    )

    # Exercise two public-API edge cases on a deliberately narrow,
    # noncommensurate grid.  BA must be exactly one Fan map from G0 even when
    # shifted points fall outside the grid, and the T=0 one-way sweep should
    # honor the configured convergence tolerances when interpolation is used.
    edge_model = ModelParams(
        omega0_meV=24.8,
        huang_rhys=0.5,
        j_meV=12.4,
        temperature_K=300.0,
        eta_meV=0.5,
    )
    edge_grid = GridSpec(1.70, 2.40, 1.65, 2.45, 0.137)
    edge_ba = scba.solve_ba(edge_model, edge_grid, controls)
    occupation = edge_model.bose_occupation
    analytic_edge_ba = edge_model.coupling_squared_eV2 * (
        (occupation + 1.0)
        * scba.bare_local_green(
            edge_ba.omega_eV - edge_model.omega0_eV, edge_model
        )
        + occupation
        * scba.bare_local_green(
            edge_ba.omega_eV + edge_model.omega0_eV, edge_model
        )
    )
    edge_ba_error = float(
        np.max(np.abs(edge_ba.sigma_eV - analytic_edge_ba))
    )
    _less_equal_check(
        checks,
        "BA_exact_one_map_narrow_nonaligned_grid_eV",
        edge_ba_error,
        5.0e-15,
    )
    nonaligned_t0 = replace(edge_model, temperature_K=0.0)
    t0_nonaligned = scba.solve_scba(nonaligned_t0, edge_grid, controls)
    _less_equal_check(
        checks,
        "SCBA_T0_nonaligned_sweep_abs_residual_eV",
        t0_nonaligned.fixed_point_abs_residual_eV,
        controls.absolute_residual_tolerance_eV,
    )
    _less_equal_check(
        checks,
        "SCBA_T0_nonaligned_sweep_relative_residual",
        t0_nonaligned.fixed_point_relative_residual,
        controls.relative_residual_tolerance,
    )
    edge_regressions = {
        "grid": asdict(edge_grid),
        "ba_exact_one_map_max_abs_error_eV": edge_ba_error,
        "T0_nonaligned_abs_residual_eV": (
            t0_nonaligned.fixed_point_abs_residual_eV
        ),
        "T0_nonaligned_relative_residual": (
            t0_nonaligned.fixed_point_relative_residual
        ),
    }

    interacting = ModelParams(
        omega0_meV=24.8,
        huang_rhys=0.5,
        j_meV=12.4,
        temperature_K=300.0,
        eta_meV=1.0,
    )
    interacting_grid = default_grid_spec(
        interacting, step_meV=0.20, padding_eV=0.55
    )
    reduced = scba.solve_scba(interacting, interacting_grid, controls)
    full = scba.solve_full_matrix_regression(interacting, reduced, controls)
    full_values = full.diagnostic_dict()
    _less_equal_check(
        checks,
        "full_reduced_symmetric_map_identity_eV",
        full.symmetric_map_identity_max_error_eV,
        1.0e-12,
    )
    _less_equal_check(
        checks,
        "full_reduced_symmetric_green_identity_per_eV",
        full.symmetric_green_identity_max_error_per_eV,
        1.0e-10,
    )
    _less_equal_check(
        checks,
        "full_asymmetric_seed_site_symmetry_relative",
        full.relative_site_sigma_difference,
        5.0e-6,
    )
    _less_equal_check(
        checks,
        "full_asymmetric_seed_bright_relative",
        full.relative_bright_green_difference,
        5.0e-6,
    )

    finite_temperature = run_finite_temperature_fc_validation(quick, checks)
    return {
        "profile": "quick" if quick else "thorough",
        "ed_suite": ed_suite,
        "scba_S_zero": {
            "model": asdict(s_zero),
            "errors": s_zero_errors,
            "ba": ba_zero.diagnostic_dict(),
            "scba": scba_zero.diagnostic_dict(),
        },
        "edge_grid_regressions": edge_regressions,
        "full_reduced_interacting": {
            "model": asdict(interacting),
            "reduced": reduced.diagnostic_dict(),
            "full": full_values,
        },
        "finite_temperature_J_zero_Bessel": finite_temperature,
    }


def _ed_case_definitions() -> List[Tuple[str, ModelParams]]:
    return [
        (
            "weak_S0p1_T0_r0",
            ModelParams(
                omega0_meV=100.0,
                huang_rhys=0.10,
                j_meV=0.0,
                temperature_K=0.0,
                eta_meV=1.0,
            ),
        ),
        (
            "intermediate_S1_T100_r1",
            ModelParams(
                omega0_meV=50.0,
                huang_rhys=1.0,
                j_meV=25.0,
                temperature_K=100.0,
                eta_meV=1.0,
            ),
        ),
        (
            "strong_S2_T300_r1",
            ModelParams(
                omega0_meV=24.8,
                huang_rhys=2.0,
                j_meV=12.4,
                temperature_K=300.0,
                eta_meV=1.0,
            ),
        ),
    ]


def _ed_cutoffs(model: ModelParams, quick: bool) -> Tuple[int, List[int]]:
    n_thermal, _ = thermal_cutoff(model, 1.0e-10)
    recommended = recommended_n_vib(model, n_thermal)
    offsets = (-6, 0, 4) if quick else (-12, -8, -4, 0, 4)
    cutoffs = sorted(
        set(max(n_thermal, recommended + offset) for offset in offsets)
    )
    if len(cutoffs) < 2:
        cutoffs.append(cutoffs[-1] + 4)
    return n_thermal, cutoffs


def run_ed_convergence(
    quick: bool,
    checks: List[Check],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for case_id, model in _ed_case_definitions():
        n_thermal, cutoffs = _ed_cutoffs(model, quick)
        grid_spec = default_grid_spec(
            model,
            step_meV=0.20 if quick else 0.10,
            padding_eV=0.75,
        )
        results = []
        for n_vib in cutoffs:
            result = ed.solve_ed(
                model,
                grid_spec,
                n_vib=n_vib,
                n_thermal=n_thermal,
                deposition_refinement=2 if quick else 4,
            )
            results.append((n_vib, result))

        reference_n, reference = results[-1]
        window = (grid_spec.analysis_min_eV, grid_spec.analysis_max_eV)
        l1_limit = 4.0e-3 if quick else 2.0e-3
        wasserstein_limit = 0.15 if quick else 0.08
        peak_limit = grid_spec.step_meV + 1.0e-9
        for n_vib, result in results:
            comparison = _compare_on_reference_grid(
                reference.energy_eV,
                reference.spectrum_bright,
                result.energy_eV,
                result.spectrum_bright,
                window,
            )
            diagnostics = result.diagnostics
            row = {
                "case_id": case_id,
                "is_reference": n_vib == reference_n,
                "S": model.huang_rhys,
                "r_2J_over_omega0": model.resonance_ratio,
                "J_meV": model.j_meV,
                "omega0_meV": model.omega0_meV,
                "temperature_K": model.temperature_K,
                "eta_meV": model.eta_meV,
                "grid_step_meV": grid_spec.step_meV,
                "n_thermal": n_thermal,
                "n_vib": n_vib,
                "reference_n_vib": reference_n,
                "thermal_omitted_probability": diagnostics[
                    "thermal_omitted_probability"
                ],
                "raw_bright_pole_weight": diagnostics["raw_bright_pole_weight"],
                "raw_bright_pole_area": diagnostics["raw_bright_pole_area"],
                "raw_bright_missing_from_retained_thermal": diagnostics[
                    "raw_bright_missing_from_retained_thermal"
                ],
                "analysis_area_bright": diagnostics["analysis_area_bright"],
            }
            row.update(comparison)
            row["thermal_omission_pass"] = bool(
                diagnostics["thermal_omitted_probability"] <= 1.0e-10
            )
            row["comparison_to_reference_pass"] = bool(
                comparison["unit_area_l1"] <= l1_limit
                and comparison["wasserstein_meV"] <= wasserstein_limit
                and abs(comparison["dominant_peak_shift_meV"]) <= peak_limit
            )
            rows.append(row)

        penultimate = rows[-2]
        _less_equal_check(
            checks,
            "ED_%s_penultimate_L1" % case_id,
            float(penultimate["unit_area_l1"]),
            l1_limit,
            "N_vib=%d versus %d" % (penultimate["n_vib"], reference_n),
        )
        _less_equal_check(
            checks,
            "ED_%s_penultimate_Wasserstein_meV" % case_id,
            float(penultimate["wasserstein_meV"]),
            wasserstein_limit,
        )
        _less_equal_check(
            checks,
            "ED_%s_penultimate_peak_shift_meV" % case_id,
            abs(float(penultimate["dominant_peak_shift_meV"])),
            peak_limit,
        )
        _less_equal_check(
            checks,
            "ED_%s_thermal_omission" % case_id,
            float(reference.diagnostics["thermal_omitted_probability"]),
            1.0e-10,
        )
        _less_equal_check(
            checks,
            "ED_%s_reference_missing_pole_weight" % case_id,
            abs(
                float(
                    reference.diagnostics[
                        "raw_bright_missing_from_retained_thermal"
                    ]
                )
            ),
            2.0e-8 if quick else 2.0e-10,
        )
    return rows


def _scba_row(
    check_type: str,
    variant: str,
    model: ModelParams,
    grid_spec: GridSpec,
    result: scba.ScbaResult,
    reference_variant: str,
    reference: scba.ScbaResult,
    reference_grid: GridSpec,
    seed_label: str,
    scan_direction: str = "",
) -> Dict[str, Any]:
    window = (reference_grid.analysis_min_eV, reference_grid.analysis_max_eV)
    comparison = _compare_on_reference_grid(
        reference.omega_eV,
        reference.spectrum_bright,
        result.omega_eV,
        result.spectrum_bright,
        window,
    )
    analysis = grid_spec.analysis_mask(result.omega_eV)
    row: Dict[str, Any] = {
        "check_type": check_type,
        "variant": variant,
        "reference_variant": reference_variant,
        "scan_direction": scan_direction,
        "seed": seed_label,
        "S": model.huang_rhys,
        "r_2J_over_omega0": model.resonance_ratio,
        "J_meV": model.j_meV,
        "omega0_meV": model.omega0_meV,
        "temperature_K": model.temperature_K,
        "eta_meV": model.eta_meV,
        "grid_step_meV": grid_spec.step_meV,
        "internal_min_eV": grid_spec.internal_min_eV,
        "internal_max_eV": grid_spec.internal_max_eV,
        "iterations": result.iterations,
        "final_mixing": result.final_mixing,
        "fixed_point_abs_residual_eV": result.fixed_point_abs_residual_eV,
        "fixed_point_relative_residual": result.fixed_point_relative_residual,
        "green_relative_change": result.green_relative_change,
        "max_positive_Im_Sigma_eV": result.max_positive_im_sigma_eV,
        "minimum_bright_spectrum_per_eV": result.minimum_bright_spectrum_per_eV,
        "minimum_dark_spectrum_per_eV": result.minimum_dark_spectrum_per_eV,
        "channel_closure_max_abs_eV": float(
            np.max(np.abs(result.sigma_channel_sum_eV - result.sigma_eV))
        ),
        "analysis_area_bright": float(
            np.trapz(
                result.spectrum_bright[analysis], result.omega_eV[analysis]
            )
        ),
    }
    row.update(comparison)
    return row


def run_scba_convergence(
    quick: bool,
    checks: List[Check],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    model = ModelParams(
        omega0_meV=24.8,
        huang_rhys=30.0 / 24.8,
        j_meV=12.4,
        temperature_K=300.0,
        eta_meV=0.5,
    )
    base_step = 0.20 if quick else 0.10
    fine_step = 0.10 if quick else 0.05
    base_grid = default_grid_spec(model, step_meV=base_step, padding_eV=0.55)
    fine_grid = default_grid_spec(model, step_meV=fine_step, padding_eV=0.55)
    wide_grid = default_grid_spec(model, step_meV=base_step, padding_eV=1.10)
    controls = scba.ScbaNumerics(max_iterations=5000)

    base = scba.solve_scba(model, base_grid, controls)
    fine = scba.solve_scba(model, fine_grid, controls)
    wide = scba.solve_scba(model, wide_grid, controls)
    mix05_controls = replace(controls, initial_mixing=0.05)
    mix12_controls = replace(controls, initial_mixing=0.12)
    mix05 = scba.solve_scba(model, base_grid, mix05_controls)
    mix12 = scba.solve_scba(model, base_grid, mix12_controls)
    zero_seed = np.zeros_like(base.omega_eV, dtype=complex)
    zero = scba.solve_scba(model, base_grid, controls, seed=zero_seed)

    rows.extend(
        [
            _scba_row(
                "grid_step_halving", "coarse", model, base_grid, base,
                "fine", fine, fine_grid, "BA"
            ),
            _scba_row(
                "grid_step_halving", "fine", model, fine_grid, fine,
                "fine", fine, fine_grid, "BA"
            ),
            _scba_row(
                "padding_widening", "standard_padding", model, base_grid, base,
                "wide_padding", wide, wide_grid, "BA"
            ),
            _scba_row(
                "padding_widening", "wide_padding", model, wide_grid, wide,
                "wide_padding", wide, wide_grid, "BA"
            ),
            _scba_row(
                "mixing", "mix_0p05", model, base_grid, mix05,
                "mix_0p08", base, base_grid, "BA"
            ),
            _scba_row(
                "mixing", "mix_0p08", model, base_grid, base,
                "mix_0p08", base, base_grid, "BA"
            ),
            _scba_row(
                "mixing", "mix_0p12", model, base_grid, mix12,
                "mix_0p08", base, base_grid, "BA"
            ),
            _scba_row(
                "seed", "BA_seed", model, base_grid, base,
                "BA_seed", base, base_grid, "BA"
            ),
            _scba_row(
                "seed", "zero_seed", model, base_grid, zero,
                "BA_seed", base, base_grid, "zero"
            ),
        ]
    )

    grid_row = rows[0]
    padding_row = rows[2]
    _less_equal_check(
        checks,
        "SCBA_grid_halving_L1",
        float(grid_row["unit_area_l1"]),
        5.0e-3 if quick else 2.5e-3,
    )
    _less_equal_check(
        checks,
        "SCBA_grid_halving_peak_shift_meV",
        abs(float(grid_row["dominant_peak_shift_meV"])),
        base_step + 1.0e-9,
    )
    _less_equal_check(
        checks,
        "SCBA_padding_widening_L1",
        float(padding_row["unit_area_l1"]),
        3.0e-4,
    )
    for row in rows[4:9]:
        _less_equal_check(
            checks,
            "SCBA_%s_%s_L1" % (row["check_type"], row["variant"]),
            float(row["unit_area_l1"]),
            2.0e-5,
        )

    # Warm-start the same signed scan from both directions.
    signed_values = (
        (-12.4, 0.0, 12.4)
        if quick
        else (-50.0, -25.0, -12.4, 0.0, 12.4, 25.0, 50.0)
    )
    scan_template = replace(model, j_meV=max(abs(value) for value in signed_values))
    scan_grid = default_grid_spec(
        scan_template, step_meV=base_step, padding_eV=0.55
    )

    def warm_scan(values: Iterable[float]) -> Dict[float, Tuple[scba.ScbaResult, str]]:
        output: Dict[float, Tuple[scba.ScbaResult, str]] = {}
        seed_result: Optional[scba.ScbaResult] = None
        for j_meV in values:
            scan_model = replace(model, j_meV=float(j_meV))
            label = "BA" if seed_result is None else "previous_J"
            result = scba.solve_scba(
                scan_model, scan_grid, controls, seed=seed_result
            )
            output[float(j_meV)] = (result, label)
            seed_result = result
        return output

    forward = warm_scan(signed_values)
    reverse = warm_scan(reversed(signed_values))
    for j_meV in signed_values:
        scan_model = replace(model, j_meV=float(j_meV))
        forward_result, forward_seed = forward[float(j_meV)]
        reverse_result, reverse_seed = reverse[float(j_meV)]
        forward_row = _scba_row(
            "scan_direction",
            "forward_J_%g" % j_meV,
            scan_model,
            scan_grid,
            forward_result,
            "reverse_J_%g" % j_meV,
            reverse_result,
            scan_grid,
            forward_seed,
            "forward",
        )
        reverse_row = _scba_row(
            "scan_direction",
            "reverse_J_%g" % j_meV,
            scan_model,
            scan_grid,
            reverse_result,
            "forward_J_%g" % j_meV,
            forward_result,
            scan_grid,
            reverse_seed,
            "reverse",
        )
        rows.extend((forward_row, reverse_row))
        _less_equal_check(
            checks,
            "SCBA_scan_direction_J_%g_L1" % j_meV,
            float(forward_row["unit_area_l1"]),
            2.0e-5,
        )

    # At T=0 the ascending recursion is an independent fixed-point reference.
    t0_model = replace(model, huang_rhys=2.0, temperature_K=0.0)
    t0_grid = default_grid_spec(t0_model, step_meV=base_step, padding_eV=0.55)
    t0_sweep = scba.solve_scba(t0_model, t0_grid, controls)
    t0_iteration = scba.solve_scba(
        t0_model,
        t0_grid,
        controls,
        use_zero_temperature_sweep=False,
    )
    t0_row_sweep = _scba_row(
        "T0_sweep_iteration", "ascending_sweep", t0_model, t0_grid, t0_sweep,
        "ascending_sweep", t0_sweep, t0_grid, "tail_recursion"
    )
    t0_row_iteration = _scba_row(
        "T0_sweep_iteration", "damped_iteration", t0_model, t0_grid,
        t0_iteration, "ascending_sweep", t0_sweep, t0_grid, "BA"
    )
    rows.extend((t0_row_sweep, t0_row_iteration))
    _less_equal_check(
        checks,
        "SCBA_T0_sweep_iteration_L1",
        float(t0_row_iteration["unit_area_l1"]),
        2.0e-6,
    )

    full = scba.solve_full_matrix_regression(model, base, controls)
    full_spectrum = -np.imag(full.green_bright_per_eV)
    full_comparison = _compare_on_reference_grid(
        base.omega_eV,
        base.spectrum_bright,
        base.omega_eV,
        full_spectrum,
        (base_grid.analysis_min_eV, base_grid.analysis_max_eV),
    )
    full_row = _scba_row(
        "full_matrix", "reduced_reference", model, base_grid, base,
        "reduced_reference", base, base_grid, "BA"
    )
    # ``_scba_row`` is initialized from an ScbaResult so that the generic
    # convergence/causality columns retain a complete schema.  Replace every
    # spectrum-comparison field with the actual asymmetric full-matrix result;
    # otherwise the row would misleadingly contain reduced-vs-reduced zeros
    # for the remaining shape, overlap, moment, and raw-area metrics.
    full_row.update(full_comparison)
    full_row.update(
        {
            "variant": "full_asymmetric_seed",
            "seed": "asymmetric_reduced_SCBA",
            "analysis_area_bright": full_comparison["candidate_raw_area"],
            "iterations": full.iterations,
            "final_mixing": full.final_mixing,
            "fixed_point_abs_residual_eV": full.fixed_point_abs_residual_eV,
            "fixed_point_relative_residual": (
                full.fixed_point_relative_residual
            ),
            "green_relative_change": full.green_relative_change,
            "max_positive_Im_Sigma_eV": float(
                max(
                    0.0,
                    np.max(full.sigma_site1_eV.imag),
                    np.max(full.sigma_site2_eV.imag),
                )
            ),
            "minimum_bright_spectrum_per_eV": float(
                np.min(full_spectrum)
            ),
            "minimum_dark_spectrum_per_eV": float(
                np.min(-np.imag(full.green_dark_per_eV))
            ),
            # A nonsymmetric full-site iterate has no unique B/D internal-line
            # channel decomposition.  Store its directly comparable closure
            # diagnostic—the final full-site Fan-map residual—and label the
            # distinction explicitly in the heterogeneous convergence table.
            "channel_closure_max_abs_eV": (
                full.fixed_point_abs_residual_eV
            ),
            "channel_closure_context": (
                "full_site_fixed_map_residual_not_reduced_BD_channels"
            ),
            "full_iterations": full.iterations,
            "full_relative_site_sigma_difference": (
                full.relative_site_sigma_difference
            ),
            "full_relative_mean_sigma_difference": (
                full.relative_mean_sigma_difference
            ),
            "full_relative_bright_green_difference": (
                full.relative_bright_green_difference
            ),
            "full_symmetric_map_identity_eV": (
                full.symmetric_map_identity_max_error_eV
            ),
            "full_symmetric_green_identity_per_eV": (
                full.symmetric_green_identity_max_error_per_eV
            ),
        }
    )
    rows.append(full_row)
    _less_equal_check(
        checks,
        "SCBA_full_matrix_bright_relative",
        full.relative_bright_green_difference,
        5.0e-6,
    )
    _less_equal_check(
        checks,
        "SCBA_full_matrix_map_identity_eV",
        full.symmetric_map_identity_max_error_eV,
        1.0e-12,
    )

    for index, row in enumerate(rows):
        causality_ok = (
            float(row["max_positive_Im_Sigma_eV"]) <= 5.0e-10
            and float(row["minimum_bright_spectrum_per_eV"]) >= -1.0e-8
            and float(row["minimum_dark_spectrum_per_eV"]) >= -1.0e-8
        )
        closure_ok = float(row["channel_closure_max_abs_eV"]) <= 1.1e-9
        row["causality_pass"] = causality_ok
        row["channel_closure_pass"] = closure_ok
        row["validation_pass"] = causality_ok and closure_ok
        if not causality_ok:
            checks.append(
                Check(
                    "SCBA_row_%d_causality" % index,
                    1.0,
                    "==",
                    0.0,
                    False,
                    "%s/%s" % (row["check_type"], row["variant"]),
                )
            )
        if not closure_ok:
            checks.append(
                Check(
                    "SCBA_row_%d_channel_closure" % index,
                    float(row["channel_closure_max_abs_eV"]),
                    "<=",
                    1.1e-9,
                    False,
                    "%s/%s" % (row["check_type"], row["variant"]),
                )
            )
    _less_equal_check(
        checks,
        "SCBA_all_rows_max_positive_Im_Sigma_eV",
        max(float(row["max_positive_Im_Sigma_eV"]) for row in rows),
        5.0e-10,
    )
    _greater_equal_check(
        checks,
        "SCBA_all_rows_minimum_spectrum_per_eV",
        min(
            min(
                float(row["minimum_bright_spectrum_per_eV"]),
                float(row["minimum_dark_spectrum_per_eV"]),
            )
            for row in rows
        ),
        -1.0e-8,
    )
    _less_equal_check(
        checks,
        "SCBA_all_rows_channel_closure_eV",
        max(float(row["channel_closure_max_abs_eV"]) for row in rows),
        1.1e-9,
    )
    return rows


class _LegacyZeroTail:
    """Temporarily reproduce the historical FigS5 finite-grid boundary."""

    def __enter__(self) -> None:
        self.original = scba.asymptotic_sigma

        def zero_tail(points: np.ndarray, model: ModelParams) -> np.ndarray:
            del model
            return np.zeros_like(np.asarray(points, dtype=float), dtype=complex)

        scba.asymptotic_sigma = zero_tail

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        scba.asymptotic_sigma = self.original


def run_figs5_regression(checks: List[Check]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with _LegacyZeroTail():
        for huang_rhys in (0.5, 1.0, 2.0):
            ba_path = ARCHIVED_FIGS5 / (
                "single_mode_single_shot_ba_S_%.1f.txt" % huang_rhys
            )
            scba_path = ARCHIVED_FIGS5 / (
                "single_mode_g_only_S_%.1f.txt" % huang_rhys
            )
            if not ba_path.is_file() or not scba_path.is_file():
                raise FileNotFoundError(
                    "Required archived former-FigS5 regression input is missing"
                )
            stored_ba = np.loadtxt(str(ba_path))
            stored_scba = np.loadtxt(str(scba_path))
            if stored_ba.shape != (10000, 5) or stored_scba.shape != (10000, 5):
                raise RuntimeError("Unexpected FigS5 regression file shape")
            if not np.array_equal(stored_ba[:, 0], stored_scba[:, 0]):
                raise RuntimeError("FigS5 BA and SCBA grids differ")
            old_grid = stored_ba[:, 0]
            step_meV = float(np.mean(np.diff(old_grid)) * 1.0e3)
            grid_spec = GridSpec(
                analysis_min_eV=float(old_grid[0]),
                analysis_max_eV=float(old_grid[-1]),
                internal_min_eV=float(old_grid[0]),
                internal_max_eV=float(old_grid[-1]),
                step_meV=step_meV,
            )
            model = ModelParams(
                energy_00_eV=2.0,
                omega0_meV=200.0,
                huang_rhys=huang_rhys,
                j_meV=0.0,
                temperature_K=0.0,
                eta_meV=1.0,
            )
            controls = scba.ScbaNumerics(
                max_iterations=2000,
                initial_mixing=0.20,
            )
            calculated_ba = scba.solve_ba(model, grid_spec, controls)
            calculated_scba = scba.solve_scba(
                model,
                grid_spec,
                controls,
                use_zero_temperature_sweep=False,
            )
            grid_error = float(
                np.max(np.abs(calculated_ba.omega_eV[:10000] - old_grid))
            )
            _less_equal_check(
                checks,
                "FigS5_S%.1f_first10000_grid_error_eV" % huang_rhys,
                grid_error,
                2.0e-12,
            )
            for method, result, stored, source in (
                ("single_shot_ba", calculated_ba, stored_ba, ba_path),
                ("scba", calculated_scba, stored_scba, scba_path),
            ):
                candidate = result.spectrum_bright[:10000]
                reference = stored[:, 1]
                comparison = compare_spectra(
                    old_grid,
                    reference,
                    candidate,
                    (float(old_grid[0]), float(old_grid[-1])),
                )
                maximum_absolute = float(np.max(np.abs(candidate - reference)))
                relative_peak = maximum_absolute / max(
                    1.0, float(np.max(np.abs(reference)))
                )
                row = {
                    "S": huang_rhys,
                    "method": method,
                    "source_file": str(source.relative_to(PROJECT_ROOT)),
                    "stored_column": 2,
                    "matched_points": 10000,
                    "grid_max_abs_error_eV": grid_error,
                    "boundary_convention": (
                        "legacy zero-Sigma outside 1.5--3.2 eV; "
                        "production SCBA retains causal first-Born tail"
                    ),
                    "stored_raw_area": comparison.reference_raw_area,
                    "calculated_raw_area": comparison.candidate_raw_area,
                    "raw_area_relative_difference": (
                        comparison.raw_area_relative_difference
                    ),
                    "maximum_absolute_difference_per_eV": maximum_absolute,
                    "maximum_relative_to_stored_peak": relative_peak,
                    "unit_area_l1": comparison.unit_area_l1,
                    "wasserstein_meV": comparison.wasserstein_eV * 1.0e3,
                    "iterations": result.iterations,
                    "fixed_point_abs_residual_eV": (
                        result.fixed_point_abs_residual_eV
                    ),
                }
                row["validation_pass"] = bool(
                    relative_peak <= 1.0e-6
                    and comparison.unit_area_l1 <= 1.0e-6
                    and grid_error <= 2.0e-12
                )
                rows.append(row)
                _less_equal_check(
                    checks,
                    "FigS5_S%.1f_%s_relative_peak" % (huang_rhys, method),
                    relative_peak,
                    1.0e-6,
                )
                _less_equal_check(
                    checks,
                    "FigS5_S%.1f_%s_unit_area_L1" % (huang_rhys, method),
                    comparison.unit_area_l1,
                    1.0e-6,
                )
    return rows


def _write_report(
    quick: bool,
    checks: Sequence[Check],
    elapsed_seconds: float,
) -> None:
    failures = [check for check in checks if not check.passed]
    lines = [
        "Single-mode Frenkel-dimer benchmark numerical validation",
        "status = %s" % ("PASS" if not failures else "FAIL"),
        "profile = %s" % ("quick" if quick else "thorough"),
        "elapsed_seconds = %.6f" % elapsed_seconds,
        "checks_total = %d" % len(checks),
        "checks_passed = %d" % (len(checks) - len(failures)),
        "checks_failed = %d" % len(failures),
        "generated_files = analytic_validation.json; ed_convergence_summary.csv; scba_convergence_summary.csv; figs5_regression.csv",
        "FigS5_boundary_note = regression alone uses the historical zero-Sigma edge; production uses the causal first-Born tail",
        "",
        "Checks:",
    ]
    for check in checks:
        lines.append(
            "%s | %s | value=%.16e %s threshold=%.16e%s"
            % (
                "PASS" if check.passed else "FAIL",
                check.name,
                check.value,
                check.comparison,
                check.threshold,
                " | " + check.detail if check.detail else "",
            )
        )
    NUMERICAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    NUMERICAL_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use fewer ED truncations and coarser reference grids.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    import time

    args = _build_parser().parse_args(argv)
    start = time.monotonic()
    ensure_directories()
    checks: List[Check] = []

    print("Running analytic and representation validations...")
    analytic_check_start = len(checks)
    analytic = run_analytic_validations(args.quick, checks)
    analytic_checks = checks[analytic_check_start:]
    analytic["validation_pass"] = all(check.passed for check in analytic_checks)
    analytic["threshold_checks"] = [asdict(check) for check in analytic_checks]
    _write_json(ANALYTIC_JSON, analytic)

    print("Running ED truncation convergence...")
    ed_rows = run_ed_convergence(args.quick, checks)
    _write_csv(ED_CONVERGENCE_CSV, ed_rows)

    print("Running SCBA numerical convergence...")
    scba_rows = run_scba_convergence(args.quick, checks)
    _write_csv(SCBA_CONVERGENCE_CSV, scba_rows)

    print("Running FigS5 matched-grid regression...")
    figs5_rows = run_figs5_regression(checks)
    _write_csv(FIGS5_REGRESSION_CSV, figs5_rows)

    elapsed = time.monotonic() - start
    _write_report(args.quick, checks, elapsed)
    failures = [check for check in checks if not check.passed]
    print(
        "%s: %d/%d checks passed in %.1f s"
        % (
            "PASS" if not failures else "FAIL",
            len(checks) - len(failures),
            len(checks),
            elapsed,
        )
    )
    print("Wrote %s" % ANALYTIC_JSON)
    print("Wrote %s" % ED_CONVERGENCE_CSV)
    print("Wrote %s" % SCBA_CONVERGENCE_CSV)
    print("Wrote %s" % FIGS5_REGRESSION_CSV)
    print("Wrote %s" % NUMERICAL_REPORT)
    if failures:
        for check in failures:
            print(
                "FAILED %s: %.6e %s %.6e"
                % (check.name, check.value, check.comparison, check.threshold)
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
