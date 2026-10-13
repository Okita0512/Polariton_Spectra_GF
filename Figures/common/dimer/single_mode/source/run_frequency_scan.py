#!/usr/bin/env python3
"""Controlled fixed-S phonon-frequency scan for the single-mode dimer.

The primary scan holds ``S``, ``|J|``, ``T`` and ``eta/omega0`` fixed and
uses ``r_abs = 2|J|/omega0`` as its independent variable.  All primary cases
share one dimensionless frequency grid.  Compact metrics and pole records are
kept for every case; full spectra are archived only for the explicitly marked
representative cases used by figures.

Outputs are isolated below ``results/frequency_scan`` and
``validation/frequency_scan``.  The manifest and result tables are replaced
atomically, so an interrupted calculation can be resumed safely::

    python3 run_frequency_scan.py --stage smoke
    python3 run_frequency_scan.py --stage primary
    python3 run_frequency_scan.py --stage secondary
    python3 run_frequency_scan.py --stage production

The implementation imports the established ED, BA, SCBA and metric APIs; it
does not duplicate either solver.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.stats import wasserstein_distance


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from calculate_single_mode_ed import (  # noqa: E402
    EDResult,
    common_mode_measure,
    difference_mode_eigensystem,
    difference_mode_measures,
    solve_ed,
)
from calculate_single_mode_scba import (  # noqa: E402
    ScbaNumerics,
    ScbaResult,
    solve_ba,
    solve_full_matrix_regression,
    solve_scba,
)
from single_mode_common import GridSpec, ModelParams, encode_metadata  # noqa: E402
from single_mode_metrics import (  # noqa: E402
    PeakAnalysis,
    PeakRecord,
    analyze_peaks,
    compare_spectra,
    match_sidebands,
    projected_self_energy_at_peak,
)


SCHEMA_VERSION = "frequency_scan_v1"
RESULT_DIR = MODULE_DIR / "results" / "frequency_scan"
SPECTRA_DIR = RESULT_DIR / "spectra"
VALIDATION_DIR = MODULE_DIR / "validation" / "frequency_scan"
MANIFEST_PATH = RESULT_DIR / "manifest.csv"
SUMMARY_PATH = RESULT_DIR / "summary.csv"
SIDEBAND_PATH = RESULT_DIR / "sideband_matches.csv"
BASELINE_PATH = RESULT_DIR / "local_baselines.csv"
# ``exact_level_map.py`` owns the authoritative parity-resolved level and
# crossing products in this directory.  This scan consumes those files but
# never overwrites them.
AUTHORITATIVE_LEVEL_PATH = RESULT_DIR / "exact_level_map.csv"
AUTHORITATIVE_GAP_PATH = RESULT_DIR / "exact_level_map_crossings.csv"
CUTOFF_PATH = VALIDATION_DIR / "ed_cutoff_convergence.csv"
GRID_CONVERGENCE_PATH = VALIDATION_DIR / "grid_padding_convergence.csv"
J0_SCALING_PATH = VALIDATION_DIR / "j0_scaling.csv"
VALIDATION_JSON = VALIDATION_DIR / "validation.json"
VALIDATION_TEXT = VALIDATION_DIR / "numerical_validation.txt"

PRIMARY_S = (0.02, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0)
J_ABS_MEV = 25.0
ETA_OVER_OMEGA = 0.01
DX = 0.002
ANALYSIS_X = (-18.0, 18.0)
INTERNAL_X = (-32.0, 24.0)
PROMINENCE_FRACTION = 0.01
MINIMUM_PEAK_DISTANCE_OVER_OMEGA = 0.02
MATCHING_RADIUS_OVER_OMEGA = 0.35

# Explicitly operational, resolution-dependent accuracy gate.  It is not a
# phase boundary and every component remains separately available in the CSV.
VALID_L1 = 0.10
VALID_W_OVER_OMEGA = 0.01
VALID_TRACKED_SHIFT_OVER_OMEGA = ETA_OVER_OMEGA
VALID_TRACKED_WEIGHT_ERROR = 0.05
VALID_SIDEBAND_WEIGHT_ERROR = 0.10
SIGNIFICANT_BASIN_WEIGHT = 0.01
VALID_RAW_AREA_RELATIVE_ERROR = 3.0e-6

CUTOFF_W_OVER_OMEGA = 2.0e-9
CUTOFF_WEIGHT_MISSING = 2.0e-12
CUTOFF_MOMENT_ERROR_OVER_OMEGA = 2.0e-10
CUTOFF_MAX = 44


@dataclass(frozen=True)
class FrequencyCase:
    case_id: str
    family: str
    model: ModelParams
    r_abs: float
    j_sign: int
    save_archive: bool = False


@dataclass(frozen=True)
class TrackInfo:
    eigen_index: int
    bare_max_index: int
    difference_energy_eV: float
    full_zero_common_energy_eV: float
    detuning_over_omega: float
    bare_bright_overlap: float
    bare_overlap_margin: float
    continuity_overlap: float
    parity_expectation: float
    full_pole_weight: float
    identity_switch: bool


MANIFEST_COLUMNS = (
    "schema_version", "case_id", "family", "status", "definition_key",
    "archive_file", "save_archive", "energy_00_eV", "omega0_meV",
    "huang_rhys", "lambda_meV", "j_meV", "j_sign", "r_abs",
    "r_signed", "temperature_K", "eta_meV", "eta_over_omega0",
    "analysis_x_min", "analysis_x_max", "internal_x_min", "internal_x_max",
    "dx", "grid_samples", "n_vib", "started_utc", "finished_utc",
    "runtime_s", "error_type", "error_message", "error_traceback",
)

SUMMARY_COLUMNS = (
    "schema_version", "case_id", "family", "energy_00_eV", "omega0_meV",
    "huang_rhys", "lambda_meV", "j_meV", "j_sign", "r_abs", "r_signed",
    "temperature_K", "eta_meV", "eta_over_omega0", "analysis_x_min",
    "analysis_x_max", "internal_x_min", "internal_x_max", "dx",
    "analysis_samples", "n_vib", "runtime_ed_s", "runtime_ba_s",
    "runtime_scba_s", "ed_raw_area", "ba_raw_area", "scba_raw_area",
    "ba_raw_area_relative_difference", "scba_raw_area_relative_difference",
    "ba_l1", "scba_l1", "ba_w_over_omega0", "scba_w_over_omega0",
    "ed_peak_count", "ba_peak_count", "scba_peak_count",
    "ba_matched_count", "ba_missing_count", "ba_spurious_count",
    "ba_significant_missing_count", "ba_significant_spurious_count",
    "ba_sideband_weight_error", "ba_matched_energy_mae_over_omega0",
    "scba_matched_count", "scba_missing_count", "scba_spurious_count",
    "scba_significant_missing_count", "scba_significant_spurious_count",
    "scba_sideband_weight_error", "scba_matched_energy_mae_over_omega0",
    "tracked_level_index", "tracked_bare_max_index",
    "tracked_index_basis",
    "tracked_difference_energy_eV", "tracked_exact_pole_energy_eV",
    "tracked_exact_detuning_over_omega0", "tracked_bare_bright_overlap",
    "tracked_bare_overlap_margin", "tracked_continuity_overlap", "tracked_parity_expectation",
    "tracked_exact_pole_weight", "tracked_identity_switch",
    "tracked_ed_peak_ordinal", "tracked_ed_peak_energy_eV",
    "tracked_ed_peak_distance_over_omega0", "tracked_ed_basin_weight",
    "tracked_ed_ambiguity", "tracked_scba_peak_ordinal",
    "tracked_scba_peak_energy_eV", "tracked_scba_shift_over_omega0",
    "tracked_scba_basin_weight", "tracked_scba_weight_error",
    "tracked_scba_ambiguity", "exact_projected_sigma_real_over_omega0",
    "exact_projected_minus_two_imag_sigma_over_omega0",
    "scba_sigma_real_over_omega0_at_tracked_peak",
    "scba_minus_two_imag_sigma_over_omega0_at_tracked_peak",
    "scba_bright_emission_minus_two_imag_over_omega0",
    "scba_dark_emission_minus_two_imag_over_omega0",
    "scba_bright_absorption_minus_two_imag_over_omega0",
    "scba_dark_absorption_minus_two_imag_over_omega0",
    "scba_channel_closure_over_omega0", "ed_pole_weight_error",
    "ed_common_omitted_weight", "ed_first_moment_error_over_omega0",
    "ed_minimum_spectrum", "scba_fixed_point_abs_residual_over_omega0",
    "scba_fixed_point_relative_residual", "scba_iterations",
    "scba_max_positive_im_sigma_over_omega0", "scba_minimum_spectrum",
    "baseline_case_id", "baseline_scba_l1", "baseline_scba_w_over_omega0",
    "resonance_excess_l1_diagnostic", "resonance_excess_w_over_omega0_diagnostic",
    "valid_l1", "valid_wasserstein", "valid_tracked_shift",
    "valid_tracked_weight", "valid_sideband_weight", "valid_peak_topology", "valid_raw_area",
    "valid_tracking", "operational_validity_pass", "archive_file",
    "significant_basin_weight_threshold",
)

SIDEBAND_COLUMNS = (
    "case_id", "family", "approximation", "status", "j_sign", "r_abs",
    "omega0_meV", "huang_rhys", "reference_ordinal", "candidate_ordinal",
    "reference_energy_eV", "candidate_energy_eV",
    "reference_detuning_over_omega0", "candidate_detuning_over_omega0",
    "energy_shift_over_omega0", "reference_weight", "candidate_weight",
    "weight_difference", "absolute_weight_error", "assignment_cost",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical(value: float) -> str:
    if abs(value) < 5.0e-15:
        value = 0.0
    return format(float(value), ".12g")


def _tag(value: float) -> str:
    return _canonical(value).replace("-", "m").replace("+", "p").replace(".", "p")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _ensure_dirs() -> None:
    for path in (RESULT_DIR, SPECTRA_DIR, VALIDATION_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=str(path.parent),
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise


def _read_csv_by_id(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["case_id"]: dict(row) for row in csv.DictReader(handle)}


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def primary_r_grid() -> Tuple[float, ...]:
    """Coarse 0.1 grid plus measured 0.005 refinement at every integer.

    Integer arithmetic in units of 0.005 keeps case IDs deterministic and
    avoids floating-point duplicates.  The refined windows support direct
    line plots of both the odd allowed guides and the even parity-forbidden
    controls without interpolation or smoothing.
    """

    # Preserve the complete command-14 mesh (0.1 globally and 0.01 within
    # +/-0.20 of odd guides) before adding the finer command-15 windows.
    two_hundredths = set(range(50, 1101, 20))
    two_hundredths.add(1100)
    for center in (200, 600, 1000):
        two_hundredths.update(range(center - 40, center + 41, 2))
    for center in (200, 400, 600, 800, 1000):
        two_hundredths.update(range(center - 30, center + 31))
    return tuple(
        value / 200.0
        for value in sorted(two_hundredths)
        if 50 <= value <= 1100
    )


def secondary_r_grid() -> Tuple[float, ...]:
    values = (
        0.25, 0.50, 0.75, 0.90, 0.95, 1.00, 1.05, 1.10, 1.25,
        1.50, 2.00, 2.50, 2.80, 2.90, 3.00, 3.10, 3.20, 3.50,
        4.00, 4.50, 4.80, 4.90, 5.00, 5.10, 5.20, 5.50,
    )
    return tuple(values)


REPRESENTATIVE_PRIMARY = {
    (0.02, 1.0, 1), (0.02, 2.0, 1), (0.02, 4.0, 1),
    (0.05, 1.0, 1), (0.05, 2.0, 1),
    # Command-15 main-text panels, selected from measured Wasserstein extrema.
    (0.05, 0.88, 1), (0.05, 1.45, 1), (0.05, 1.94, 1),
    (0.05, 0.88, -1),
    # Command-16 first-odd sweep: lower flank, measured maximum, bare guide,
    # and upper off-integer reference minimum for S=0.05 and S=0.10.
    (0.05, 0.65, 1),
    (0.10, 0.65, 1), (0.10, 0.80, 1), (0.10, 1.00, 1),
    (0.10, 1.35, 1),
    (0.50, 1.0, 1), (0.50, 2.0, 1), (0.50, 3.0, 1), (0.50, 4.0, 1),
    (1.00, 3.0, 1),
    (0.02, 1.0, -1), (0.05, 1.0, -1), (0.50, 1.0, -1), (1.00, 1.0, -1),
}


def _model_for_r(s_value: float, r_abs: float, sign: int, fixed_lambda: bool = False) -> ModelParams:
    omega0 = 2.0 * J_ABS_MEV / r_abs
    s_model = 30.0 / omega0 if fixed_lambda else s_value
    model = ModelParams(
        energy_00_eV=2.0,
        omega0_meV=omega0,
        huang_rhys=s_model,
        j_meV=float(sign) * J_ABS_MEV,
        temperature_K=0.0,
        eta_meV=ETA_OVER_OMEGA * omega0,
    )
    model.validate()
    return model


def build_cases() -> List[FrequencyCase]:
    cases: List[FrequencyCase] = []
    for s_value in PRIMARY_S:
        for sign in (-1, 1):
            for r_abs in primary_r_grid():
                model = _model_for_r(s_value, r_abs, sign)
                case_id = "primary_S%s_J%s_r%s" % (
                    _tag(s_value), "p" if sign > 0 else "m", _tag(r_abs)
                )
                cases.append(FrequencyCase(
                    case_id, "primary_fixed_S", model, r_abs, sign,
                    (s_value, r_abs, sign) in REPRESENTATIVE_PRIMARY,
                ))
    # One dimensionless baseline per S; explicit scale invariance is validated
    # at additional frequencies rather than duplicating identical spectra.
    for s_value in PRIMARY_S:
        model = ModelParams(
            energy_00_eV=2.0, omega0_meV=50.0, huang_rhys=s_value,
            j_meV=0.0, temperature_K=0.0, eta_meV=0.5,
        )
        cases.append(FrequencyCase(
            "baseline_S%s_w50" % _tag(s_value), "primary_J0_baseline",
            model, 0.0, 0, s_value in (0.05, 0.50, 2.0),
        ))

    for r_abs in secondary_r_grid():
        model0 = _model_for_r(0.0, r_abs, 1, fixed_lambda=True)
        s_value = model0.huang_rhys
        for sign in (-1, 1):
            model = _model_for_r(0.0, r_abs, sign, fixed_lambda=True)
            cases.append(FrequencyCase(
                "fixedlambda_J%s_r%s" % ("p" if sign > 0 else "m", _tag(r_abs)),
                "secondary_fixed_lambda", model, r_abs, sign, False,
            ))
        baseline = ModelParams(
            energy_00_eV=2.0, omega0_meV=model0.omega0_meV,
            huang_rhys=s_value, j_meV=0.0, temperature_K=0.0,
            eta_meV=model0.eta_meV,
        )
        cases.append(FrequencyCase(
            "fixedlambda_baseline_r%s" % _tag(r_abs),
            "secondary_J0_baseline", baseline, r_abs, 0, False,
        ))
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Frequency-scan case ID collision")
    return cases


def dimensionless_grid(model: ModelParams, dx: float = DX,
                       analysis_x: Tuple[float, float] = ANALYSIS_X,
                       internal_x: Tuple[float, float] = INTERNAL_X) -> GridSpec:
    if not (internal_x[0] < analysis_x[0] < analysis_x[1] < internal_x[1]):
        raise ValueError("Dimensionless grid bounds are not nested")
    bright = model.vertical_energy_eV + model.j_eV
    omega = model.omega0_eV
    return GridSpec(
        analysis_min_eV=bright + analysis_x[0] * omega,
        analysis_max_eV=bright + analysis_x[1] * omega,
        internal_min_eV=bright + internal_x[0] * omega,
        internal_max_eV=bright + internal_x[1] * omega,
        step_meV=dx * model.omega0_meV,
    )


def _definition_key(case: FrequencyCase, grid: GridSpec) -> str:
    payload = {
        "schema": SCHEMA_VERSION,
        "model": asdict(case.model),
        "grid": asdict(grid),
        "save_archive": case.save_archive,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _archive_path(case: FrequencyCase) -> Path:
    return SPECTRA_DIR / (case.case_id + ".npz")


def _manifest_base(case: FrequencyCase, n_vib: int) -> Dict[str, Any]:
    model = case.model
    grid = dimensionless_grid(model)
    archive = _archive_path(case) if case.save_archive else None
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case.case_id,
        "family": case.family,
        "status": "pending",
        "definition_key": _definition_key(case, grid),
        "archive_file": "" if archive is None else str(archive.relative_to(MODULE_DIR)),
        "save_archive": str(case.save_archive).lower(),
        "energy_00_eV": model.energy_00_eV,
        "omega0_meV": model.omega0_meV,
        "huang_rhys": model.huang_rhys,
        "lambda_meV": model.reorganization_eV * 1000.0,
        "j_meV": model.j_meV,
        "j_sign": case.j_sign,
        "r_abs": case.r_abs,
        "r_signed": model.resonance_ratio,
        "temperature_K": model.temperature_K,
        "eta_meV": model.eta_meV,
        "eta_over_omega0": model.eta_eV / model.omega0_eV,
        "analysis_x_min": ANALYSIS_X[0], "analysis_x_max": ANALYSIS_X[1],
        "internal_x_min": INTERNAL_X[0], "internal_x_max": INTERNAL_X[1],
        "dx": DX, "grid_samples": grid.grid().size, "n_vib": n_vib,
        "started_utc": "", "finished_utc": "", "runtime_s": "",
        "error_type": "", "error_message": "", "error_traceback": "",
    }


def _combined_bright_sticks(model: ModelParams, n_vib: int) -> Tuple[np.ndarray, np.ndarray]:
    probabilities = np.ones(1, dtype=float)
    common = common_mode_measure(model, n_vib, probabilities)
    difference, _, _ = difference_mode_measures(model, n_vib, probabilities)
    energies = (common.energy_eV[:, None] + difference.energy_eV[None, :]).ravel()
    weights = (common.weight[:, None] * difference.weight[None, :]).ravel()
    return energies, weights


def _cutoff_comparison(model: ModelParams, low: int, high: int) -> Dict[str, float]:
    energy_low, weight_low = _combined_bright_sticks(model, low)
    energy_high, weight_high = _combined_bright_sticks(model, high)
    omega = model.omega0_eV
    w_distance = wasserstein_distance(
        energy_low, energy_high, u_weights=weight_low, v_weights=weight_high
    ) / omega
    expected_moment = model.vertical_energy_eV + model.j_eV
    moment = float(np.sum(energy_high * weight_high) / np.sum(weight_high))
    return {
        "stick_wasserstein_over_omega0": float(w_distance),
        "maximum_weight_missing": float(max(
            abs(1.0 - np.sum(weight_low)), abs(1.0 - np.sum(weight_high))
        )),
        "first_moment_error_over_omega0": float(abs(moment - expected_moment) / omega),
    }


def _cutoff_pass(values: Mapping[str, float]) -> bool:
    return (
        values["stick_wasserstein_over_omega0"] <= CUTOFF_W_OVER_OMEGA
        and values["maximum_weight_missing"] <= CUTOFF_WEIGHT_MISSING
        and values["first_moment_error_over_omega0"] <= CUTOFF_MOMENT_ERROR_OVER_OMEGA
    )


def determine_primary_cutoffs() -> Tuple[Dict[float, int], List[Dict[str, Any]]]:
    """Adapt one converged cutoff per S over signs and integer controls."""

    cutoffs: Dict[float, int] = {}
    rows: List[Dict[str, Any]] = []
    test_r = (0.25, 1.0, 2.0, 3.0, 4.0, 5.0, 5.5)
    for s_value in PRIMARY_S:
        low = max(8, int(math.ceil(s_value + 5.0 * math.sqrt(s_value + 1.0))))
        if low % 2:
            low += 1
        while low + 2 <= CUTOFF_MAX:
            comparisons = []
            for sign in (-1, 1):
                for r_abs in test_r:
                    model = _model_for_r(s_value, r_abs, sign)
                    comparisons.append(_cutoff_comparison(model, low, low + 2))
            aggregate = {
                "stick_wasserstein_over_omega0": max(
                    item["stick_wasserstein_over_omega0"] for item in comparisons
                ),
                "maximum_weight_missing": max(
                    item["maximum_weight_missing"] for item in comparisons
                ),
                "first_moment_error_over_omega0": max(
                    item["first_moment_error_over_omega0"] for item in comparisons
                ),
            }
            passed = _cutoff_pass(aggregate)
            rows.append({
                "family": "primary_fixed_S", "huang_rhys": s_value,
                "n_low": low, "n_high": low + 2, **aggregate,
                "pass": passed,
            })
            if passed:
                cutoffs[s_value] = low + 2
                break
            low += 2
        if s_value not in cutoffs:
            raise RuntimeError("ED cutoff did not converge for S={}".format(s_value))
    return cutoffs, rows


def determine_secondary_cutoff(case: FrequencyCase) -> Tuple[int, List[Dict[str, Any]]]:
    s_value = case.model.huang_rhys
    low = max(10, int(math.ceil(s_value + 6.0 * math.sqrt(s_value + 1.0))))
    if low % 2:
        low += 1
    rows: List[Dict[str, Any]] = []
    while low + 2 <= CUTOFF_MAX:
        values = _cutoff_comparison(case.model, low, low + 2)
        passed = _cutoff_pass(values)
        rows.append({
            "family": case.family, "case_id": case.case_id,
            "huang_rhys": s_value, "r_abs": case.r_abs,
            "n_low": low, "n_high": low + 2, **values, "pass": passed,
        })
        if passed:
            return low + 2, rows
        low += 2
    raise RuntimeError("Secondary ED cutoff did not converge: {}".format(case.case_id))


def _parity_diagonal(n_vib: int) -> np.ndarray:
    oscillator = (-1.0) ** np.arange(n_vib, dtype=float)
    return np.concatenate((oscillator, -oscillator))


def _positive_parity_eigensystem(
    model: ModelParams, n_vib: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Diagonalize only the Pi=+1 block containing ``|B,0>``.

    Resolving the block before diagonalization prevents a general Hermitian
    eigensolver from arbitrarily rotating exactly degenerate states belonging
    to opposite parity sectors at the even controls.
    """

    number = np.diag(np.arange(n_vib, dtype=float))
    raising = np.diag(np.sqrt(np.arange(1, n_vib, dtype=float)), 1)
    coordinate = raising + raising.T
    identity_v = np.eye(n_vib)
    tau_z = np.diag([1.0, -1.0])
    tau_x = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    hamiltonian = (
        np.kron(
            model.vertical_energy_eV * np.eye(2) + model.j_eV * tau_z,
            identity_v,
        )
        + np.kron(np.eye(2), model.omega0_eV * number)
        + model.coupling_eV / math.sqrt(2.0) * np.kron(tau_x, coordinate)
    )
    indices = np.flatnonzero(_parity_diagonal(n_vib) > 0.0)
    values, sector_vectors = eigh(
        hamiltonian[np.ix_(indices, indices)],
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    vectors = np.zeros((2 * n_vib, len(values)), dtype=float)
    vectors[indices, :] = sector_vectors
    return values, vectors


def build_track_map(cases: Sequence[FrequencyCase], cutoffs: Mapping[float, int]) -> Dict[str, TrackInfo]:
    tracks: Dict[str, TrackInfo] = {}
    for s_value in PRIMARY_S:
        n_vib = cutoffs[s_value]
        parity = _parity_diagonal(n_vib)
        for sign in (-1, 1):
            group = sorted(
                (case for case in cases if case.family == "primary_fixed_S"
                 and case.j_sign == sign
                 and abs(case.model.huang_rhys - s_value) < 1.0e-12),
                key=lambda case: case.r_abs,
            )
            previous: Optional[np.ndarray] = None
            for case in group:
                eigenvalues, eigenvectors = _positive_parity_eigensystem(case.model, n_vib)
                bare_weights = np.square(np.abs(eigenvectors[0, :]))
                bare_max = int(np.argmax(bare_weights))
                two_largest = np.sort(bare_weights)[-2:]
                bare_margin = float(two_largest[-1] - two_largest[-2])
                if previous is None:
                    overlaps = np.zeros_like(bare_weights)
                    selected = bare_max
                    continuity = float(bare_weights[selected])
                else:
                    overlaps = np.square(np.abs(previous.conjugate() @ eigenvectors))
                    selected = int(np.argmax(bare_weights + 0.05 * overlaps))
                    continuity = float(overlaps[selected])
                vector = eigenvectors[:, selected]
                parity_expectation = float(np.sum(parity * np.square(np.abs(vector))))
                full_energy = float(
                    eigenvalues[selected] - 0.5 * case.model.reorganization_eV
                )
                bright_energy = case.model.vertical_energy_eV + case.model.j_eV
                full_weight = float(
                    math.exp(-0.5 * s_value) * bare_weights[selected]
                )
                tracks[case.case_id] = TrackInfo(
                    eigen_index=selected,
                    bare_max_index=bare_max,
                    difference_energy_eV=float(eigenvalues[selected]),
                    full_zero_common_energy_eV=full_energy,
                    detuning_over_omega=(full_energy - bright_energy) / case.model.omega0_eV,
                    bare_bright_overlap=float(bare_weights[selected]),
                    bare_overlap_margin=bare_margin,
                    continuity_overlap=continuity,
                    parity_expectation=parity_expectation,
                    full_pole_weight=full_weight,
                    identity_switch=bool(previous is not None and continuity < 0.50),
                )
                previous = vector.copy()
    return tracks


def _baseline_track(case: FrequencyCase, n_vib: int) -> TrackInfo:
    eigenvalues, eigenvectors = _positive_parity_eigensystem(case.model, n_vib)
    weights = np.square(np.abs(eigenvectors[0, :]))
    selected = int(np.argmax(weights))
    full_energy = float(eigenvalues[selected] - 0.5 * case.model.reorganization_eV)
    bright = case.model.vertical_energy_eV + case.model.j_eV
    parity = _parity_diagonal(n_vib)
    vector = eigenvectors[:, selected]
    return TrackInfo(
        selected, selected, float(eigenvalues[selected]), full_energy,
        (full_energy - bright) / case.model.omega0_eV,
        float(weights[selected]), float(np.sort(weights)[-1] - np.sort(weights)[-2]), 1.0,
        float(np.sum(parity * np.square(np.abs(vector)))),
        float(math.exp(-0.5 * case.model.huang_rhys) * weights[selected]), False,
    )


def _nearest_peak(peaks: PeakAnalysis, target_eV: float, omega_eV: float) -> Tuple[Optional[PeakRecord], float, str]:
    if not peaks.peaks:
        return None, float("nan"), "no_resolved_peak"
    ordered = sorted(peaks.peaks, key=lambda peak: abs(peak.energy_eV - target_eV))
    selected = ordered[0]
    distance = abs(selected.energy_eV - target_eV)
    reasons: List[str] = []
    if len(ordered) > 1:
        first = abs(ordered[0].energy_eV - target_eV)
        second = abs(ordered[1].energy_eV - target_eV)
        if second - first < 0.03 * omega_eV:
            reasons.append("nearby_competing_peak")
    return selected, float(distance), ";".join(reasons)


def _complex_at(energy: np.ndarray, values: np.ndarray, target: float) -> complex:
    return complex(
        np.interp(target, energy, values.real),
        np.interp(target, energy, values.imag),
    )


def _peak_match_for_track(
    ed_peak: Optional[PeakRecord], candidate: PeakAnalysis, omega: float
) -> Tuple[Optional[PeakRecord], str]:
    if ed_peak is None or not candidate.peaks:
        return None, "no_reference_or_candidate_peak"
    ordered = sorted(candidate.peaks, key=lambda peak: abs(peak.energy_eV - ed_peak.energy_eV))
    chosen = ordered[0]
    distance = abs(chosen.energy_eV - ed_peak.energy_eV)
    reasons: List[str] = []
    if distance > MATCHING_RADIUS_OVER_OMEGA * omega:
        return None, "no_candidate_within_matching_radius"
    if len(ordered) > 1:
        d2 = abs(ordered[1].energy_eV - ed_peak.energy_eV)
        if d2 - distance < 0.03 * omega:
            reasons.append("competing_candidate_peak")
    return chosen, ";".join(reasons)


def _sideband_rows(
    case: FrequencyCase,
    approximation: str,
    matching: Any,
    bright_energy: float,
) -> List[Dict[str, Any]]:
    omega = case.model.omega0_eV
    rows = []
    for item in matching.rows:
        ref_detuning = (
            (item.reference_energy_eV - bright_energy) / omega
            if np.isfinite(item.reference_energy_eV) else float("nan")
        )
        cand_detuning = (
            (item.candidate_energy_eV - bright_energy) / omega
            if np.isfinite(item.candidate_energy_eV) else float("nan")
        )
        rows.append({
            "case_id": case.case_id, "family": case.family,
            "approximation": approximation.upper(), "status": item.status,
            "j_sign": case.j_sign, "r_abs": case.r_abs,
            "omega0_meV": case.model.omega0_meV,
            "huang_rhys": case.model.huang_rhys,
            "reference_ordinal": item.reference_ordinal,
            "candidate_ordinal": item.candidate_ordinal,
            "reference_energy_eV": item.reference_energy_eV,
            "candidate_energy_eV": item.candidate_energy_eV,
            "reference_detuning_over_omega0": ref_detuning,
            "candidate_detuning_over_omega0": cand_detuning,
            "energy_shift_over_omega0": (
                item.energy_shift_candidate_minus_reference_eV / omega
                if np.isfinite(item.energy_shift_candidate_minus_reference_eV)
                else float("nan")
            ),
            "reference_weight": item.reference_weight,
            "candidate_weight": item.candidate_weight,
            "weight_difference": item.weight_difference_candidate_minus_reference,
            "absolute_weight_error": item.absolute_weight_error,
            "assignment_cost": item.assignment_cost,
        })
    return rows


def _save_archive(
    path: Path, case: FrequencyCase, grid: GridSpec, n_vib: int,
    track: TrackInfo, ed: EDResult, ba: ScbaResult, scba: ScbaResult,
    summary: Mapping[str, Any],
) -> None:
    energy = ed.energy_eV
    mask = grid.analysis_mask(energy)
    selected = np.flatnonzero(mask)
    saved = slice(max(0, selected[0] - 1), min(energy.size, selected[-1] + 2))
    arrays: Dict[str, np.ndarray] = {
        "energy_eV": np.asarray(energy[saved], dtype=np.float64),
        "analysis_mask": np.asarray(mask[saved], dtype=bool),
        "ed_green_bright_per_eV": np.asarray(ed.green_bright[saved], dtype=np.complex64),
        "ba_green_bright_per_eV": np.asarray(ba.green_bright_per_eV[saved], dtype=np.complex64),
        "scba_green_bright_per_eV": np.asarray(scba.green_bright_per_eV[saved], dtype=np.complex64),
        "scba_sigma_eV": np.asarray(scba.sigma_eV[saved], dtype=np.complex64),
        "scba_sigma_from_bright_emission_eV": np.asarray(scba.sigma_from_bright_emission_eV[saved], dtype=np.complex64),
        "scba_sigma_from_bright_absorption_eV": np.asarray(scba.sigma_from_bright_absorption_eV[saved], dtype=np.complex64),
        "scba_sigma_from_dark_emission_eV": np.asarray(scba.sigma_from_dark_emission_eV[saved], dtype=np.complex64),
        "scba_sigma_from_dark_absorption_eV": np.asarray(scba.sigma_from_dark_absorption_eV[saved], dtype=np.complex64),
    }
    metadata = {
        "schema_version": SCHEMA_VERSION, "case_id": case.case_id,
        "family": case.family, "model": asdict(case.model), "grid": asdict(grid),
        "dimensionless_grid": {"analysis_x": ANALYSIS_X, "internal_x": INTERNAL_X, "dx": DX},
        "n_vib": n_vib, "track": asdict(track), "summary": dict(summary),
        "spectrum_convention": "A(E)=-Im G^R(E); full-line raw area pi",
        "eta_note": "common retarded HWHM, not phonon damping",
    }
    arrays["metadata_json"] = encode_metadata(_jsonable(metadata))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=str(path.parent),
                                         prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as handle:
            temporary_name = handle.name
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        if temporary_name:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise


def run_case(
    case: FrequencyCase,
    n_vib: int,
    track: TrackInfo,
    grid_dx: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one case with an optional explicit dimensionless grid spacing.

    The override prevents profile generators from relying on a default
    argument that was bound when this module was imported.  Historical calls
    that omit it continue to use this module's ``DX``.
    """

    model = case.model
    omega = model.omega0_eV
    selected_dx = DX if grid_dx is None else float(grid_dx)
    grid = dimensionless_grid(model, dx=selected_dx)
    numerics = ScbaNumerics()
    start = time.perf_counter()
    t0 = time.perf_counter()
    ed = solve_ed(model, grid, n_vib=n_vib, n_thermal=1,
                  thermal_tolerance=1.0e-12, stick_weight_tolerance=1.0e-14,
                  deposition_refinement=4)
    runtime_ed = time.perf_counter() - t0
    t0 = time.perf_counter()
    ba = solve_ba(model, grid, numerics)
    runtime_ba = time.perf_counter() - t0
    t0 = time.perf_counter()
    scba = solve_scba(model, grid, numerics, seed=None,
                      use_zero_temperature_sweep=True, verbose=False)
    runtime_scba = time.perf_counter() - t0

    energy = ed.energy_eV
    window = (grid.analysis_min_eV, grid.analysis_max_eV)
    spectra = {
        "ed": ed.spectrum_bright,
        "ba": ba.spectrum_bright_per_eV,
        "scba": scba.spectrum_bright_per_eV,
    }
    peak_analyses = {
        name: analyze_peaks(
            energy, spectrum, window=window,
            prominence_fraction=PROMINENCE_FRACTION,
            minimum_distance_eV=MINIMUM_PEAK_DISTANCE_OVER_OMEGA * omega,
            negative_tolerance=1.0e-10,
        )
        for name, spectrum in spectra.items()
    }
    comparisons = {
        name: compare_spectra(
            energy, spectra["ed"], spectra[name], window=window,
            negative_tolerance=1.0e-10,
        )
        for name in ("ba", "scba")
    }
    matches = {
        name: match_sidebands(
            peak_analyses["ed"].peaks, peak_analyses[name].peaks,
            maximum_energy_difference_eV=MATCHING_RADIUS_OVER_OMEGA * omega,
            method="monotone",
        )
        for name in ("ba", "scba")
    }

    ed_peak, ed_peak_distance, ed_ambiguity = _nearest_peak(
        peak_analyses["ed"], track.full_zero_common_energy_eV, omega
    )
    if np.isfinite(ed_peak_distance) and ed_peak_distance > 0.08 * omega:
        ed_ambiguity = ";".join(filter(None, (ed_ambiguity, "tracked_pole_not_peak_center")))
    scba_peak, scba_ambiguity = _peak_match_for_track(
        ed_peak, peak_analyses["scba"], omega
    )
    if track.identity_switch:
        ed_ambiguity = ";".join(filter(None, (ed_ambiguity, "bright_branch_identity_switch")))
    if track.bare_overlap_margin < 0.05:
        ed_ambiguity = ";".join(filter(None, (ed_ambiguity, "competing_bare_B0_overlaps")))

    ed_peak_energy = float("nan") if ed_peak is None else ed_peak.energy_eV
    ed_peak_weight = float("nan") if ed_peak is None else ed_peak.basin_weight_fraction
    scba_peak_energy = float("nan") if scba_peak is None else scba_peak.energy_eV
    scba_peak_weight = float("nan") if scba_peak is None else scba_peak.basin_weight_fraction
    tracked_shift = (
        (scba_peak_energy - ed_peak_energy) / omega
        if np.isfinite(scba_peak_energy) and np.isfinite(ed_peak_energy)
        else float("nan")
    )
    tracked_weight_error = (
        scba_peak_weight - ed_peak_weight
        if np.isfinite(scba_peak_weight) and np.isfinite(ed_peak_weight)
        else float("nan")
    )
    if np.isfinite(ed_peak_weight) and ed_peak_weight < 0.01:
        ed_ambiguity = ";".join(filter(None, (ed_ambiguity, "tracked_ED_basin_below_one_percent")))
    if np.isfinite(scba_peak_weight) and scba_peak_weight < 0.01:
        scba_ambiguity = ";".join(filter(None, (scba_ambiguity, "tracked_SCBA_basin_below_one_percent")))

    exact_sigma = projected_self_energy_at_peak(
        energy, ed.green_bright,
        model.vertical_energy_eV + model.j_eV, model.eta_eV,
        track.full_zero_common_energy_eV,
    )
    channel_target = scba_peak_energy if np.isfinite(scba_peak_energy) else track.full_zero_common_energy_eV
    sigma_value = _complex_at(energy, scba.sigma_eV, channel_target)
    channel_arrays = {
        "bright_emission": scba.sigma_from_bright_emission_eV,
        "bright_absorption": scba.sigma_from_bright_absorption_eV,
        "dark_emission": scba.sigma_from_dark_emission_eV,
        "dark_absorption": scba.sigma_from_dark_absorption_eV,
    }
    channel_values = {
        key: _complex_at(energy, values, channel_target)
        for key, values in channel_arrays.items()
    }
    channel_closure = abs(sum(channel_values.values()) - sigma_value) / omega
    diagnostic = ed.diagnostics
    bright_energy = model.vertical_energy_eV + model.j_eV
    archive_file = str(_archive_path(case).relative_to(MODULE_DIR)) if case.save_archive else ""

    row: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "case_id": case.case_id,
        "family": case.family, "energy_00_eV": model.energy_00_eV,
        "omega0_meV": model.omega0_meV, "huang_rhys": model.huang_rhys,
        "lambda_meV": 1000.0 * model.reorganization_eV,
        "j_meV": model.j_meV, "j_sign": case.j_sign, "r_abs": case.r_abs,
        "r_signed": model.resonance_ratio, "temperature_K": model.temperature_K,
        "eta_meV": model.eta_meV, "eta_over_omega0": model.eta_eV / omega,
        "analysis_x_min": ANALYSIS_X[0], "analysis_x_max": ANALYSIS_X[1],
        "internal_x_min": INTERNAL_X[0], "internal_x_max": INTERNAL_X[1],
        "dx": DX, "analysis_samples": int(np.count_nonzero(grid.analysis_mask(energy))),
        "n_vib": n_vib, "runtime_ed_s": runtime_ed, "runtime_ba_s": runtime_ba,
        "runtime_scba_s": runtime_scba,
        "ed_raw_area": comparisons["scba"].reference_raw_area,
        "ba_raw_area": comparisons["ba"].candidate_raw_area,
        "scba_raw_area": comparisons["scba"].candidate_raw_area,
        "ba_raw_area_relative_difference": comparisons["ba"].raw_area_relative_difference,
        "scba_raw_area_relative_difference": comparisons["scba"].raw_area_relative_difference,
        "ba_l1": comparisons["ba"].unit_area_l1,
        "scba_l1": comparisons["scba"].unit_area_l1,
        "ba_w_over_omega0": comparisons["ba"].wasserstein_eV / omega,
        "scba_w_over_omega0": comparisons["scba"].wasserstein_eV / omega,
        "ed_peak_count": len(peak_analyses["ed"].peaks),
        "ba_peak_count": len(peak_analyses["ba"].peaks),
        "scba_peak_count": len(peak_analyses["scba"].peaks),
        "tracked_level_index": track.eigen_index,
        "tracked_bare_max_index": track.bare_max_index,
        "tracked_index_basis": "Pi=+1 parity-sector energy order",
        "tracked_difference_energy_eV": track.difference_energy_eV,
        "tracked_exact_pole_energy_eV": track.full_zero_common_energy_eV,
        "tracked_exact_detuning_over_omega0": track.detuning_over_omega,
        "tracked_bare_bright_overlap": track.bare_bright_overlap,
        "tracked_bare_overlap_margin": track.bare_overlap_margin,
        "tracked_continuity_overlap": track.continuity_overlap,
        "tracked_parity_expectation": track.parity_expectation,
        "tracked_exact_pole_weight": track.full_pole_weight,
        "tracked_identity_switch": track.identity_switch,
        "tracked_ed_peak_ordinal": 0 if ed_peak is None else ed_peak.ordinal,
        "tracked_ed_peak_energy_eV": ed_peak_energy,
        "tracked_ed_peak_distance_over_omega0": ed_peak_distance / omega,
        "tracked_ed_basin_weight": ed_peak_weight,
        "tracked_ed_ambiguity": ed_ambiguity,
        "tracked_scba_peak_ordinal": 0 if scba_peak is None else scba_peak.ordinal,
        "tracked_scba_peak_energy_eV": scba_peak_energy,
        "tracked_scba_shift_over_omega0": tracked_shift,
        "tracked_scba_basin_weight": scba_peak_weight,
        "tracked_scba_weight_error": tracked_weight_error,
        "tracked_scba_ambiguity": scba_ambiguity,
        "exact_projected_sigma_real_over_omega0": exact_sigma.sigma_real_eV / omega,
        "exact_projected_minus_two_imag_sigma_over_omega0": exact_sigma.minus_two_imag_sigma_eV / omega,
        "scba_sigma_real_over_omega0_at_tracked_peak": sigma_value.real / omega,
        "scba_minus_two_imag_sigma_over_omega0_at_tracked_peak": -2.0 * sigma_value.imag / omega,
        "scba_bright_emission_minus_two_imag_over_omega0": -2.0 * channel_values["bright_emission"].imag / omega,
        "scba_dark_emission_minus_two_imag_over_omega0": -2.0 * channel_values["dark_emission"].imag / omega,
        "scba_bright_absorption_minus_two_imag_over_omega0": -2.0 * channel_values["bright_absorption"].imag / omega,
        "scba_dark_absorption_minus_two_imag_over_omega0": -2.0 * channel_values["dark_absorption"].imag / omega,
        "scba_channel_closure_over_omega0": channel_closure,
        "ed_pole_weight_error": abs(1.0 - float(diagnostic["raw_bright_pole_weight"])),
        "ed_common_omitted_weight": diagnostic["common_final_basis_omitted_weight"],
        "ed_first_moment_error_over_omega0": abs(diagnostic["bright_first_moment_error_eV"]) / omega,
        "ed_minimum_spectrum": diagnostic["minimum_bright_spectrum"],
        "scba_fixed_point_abs_residual_over_omega0": scba.fixed_point_abs_residual_eV / omega,
        "scba_fixed_point_relative_residual": scba.fixed_point_relative_residual,
        "scba_iterations": scba.iterations,
        "scba_max_positive_im_sigma_over_omega0": scba.max_positive_im_sigma_eV / omega,
        "scba_minimum_spectrum": scba.minimum_bright_spectrum_per_eV,
        "baseline_case_id": "", "baseline_scba_l1": "",
        "baseline_scba_w_over_omega0": "", "resonance_excess_l1_diagnostic": "",
        "resonance_excess_w_over_omega0_diagnostic": "", "archive_file": archive_file,
    }
    all_sidebands: List[Dict[str, Any]] = []
    for name in ("ba", "scba"):
        matching = matches[name]
        significant_missing = sum(
            item.status == "missing" and item.reference_weight >= SIGNIFICANT_BASIN_WEIGHT
            for item in matching.rows
        )
        significant_spurious = sum(
            item.status == "spurious" and item.candidate_weight >= SIGNIFICANT_BASIN_WEIGHT
            for item in matching.rows
        )
        row.update({
            name + "_matched_count": matching.matched_count,
            name + "_missing_count": matching.missing_count,
            name + "_spurious_count": matching.spurious_count,
            name + "_significant_missing_count": significant_missing,
            name + "_significant_spurious_count": significant_spurious,
            name + "_sideband_weight_error": matching.aggregate_relative_weight_error,
            name + "_matched_energy_mae_over_omega0": matching.matched_energy_mae_eV / omega,
        })
        all_sidebands.extend(_sideband_rows(case, name, matching, bright_energy))
    row["significant_basin_weight_threshold"] = SIGNIFICANT_BASIN_WEIGHT

    row["valid_l1"] = bool(row["scba_l1"] <= VALID_L1)
    row["valid_wasserstein"] = bool(row["scba_w_over_omega0"] <= VALID_W_OVER_OMEGA)
    row["valid_tracked_shift"] = bool(
        np.isfinite(tracked_shift) and abs(tracked_shift) <= VALID_TRACKED_SHIFT_OVER_OMEGA
    )
    row["valid_tracked_weight"] = bool(
        np.isfinite(tracked_weight_error) and abs(tracked_weight_error) <= VALID_TRACKED_WEIGHT_ERROR
    )
    row["valid_sideband_weight"] = bool(
        np.isfinite(matches["scba"].aggregate_relative_weight_error)
        and matches["scba"].aggregate_relative_weight_error <= VALID_SIDEBAND_WEIGHT_ERROR
    )
    row["valid_peak_topology"] = bool(
        row["scba_significant_missing_count"] == 0
        and row["scba_significant_spurious_count"] == 0
    )
    row["valid_raw_area"] = bool(
        abs(row["scba_raw_area_relative_difference"]) <= VALID_RAW_AREA_RELATIVE_ERROR
    )
    row["valid_tracking"] = bool(not ed_ambiguity and not scba_ambiguity)
    row["operational_validity_pass"] = bool(all(row[key] for key in (
        "valid_l1", "valid_wasserstein", "valid_tracked_shift",
        "valid_tracked_weight", "valid_sideband_weight", "valid_peak_topology", "valid_raw_area",
        "valid_tracking",
    )))

    if case.save_archive:
        _save_archive(_archive_path(case), case, grid, n_vib, track, ed, ba, scba, row)
    row["runtime_total_s"] = time.perf_counter() - start
    return row, all_sidebands


def _archive_complete(case: FrequencyCase) -> bool:
    if not case.save_archive:
        return True
    path = _archive_path(case)
    if not path.is_file():
        return False
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            return metadata.get("schema_version") == SCHEMA_VERSION and metadata.get("case_id") == case.case_id
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _refresh_operational_validity(
    summary: Mapping[str, Dict[str, Any]],
    sidebands: Sequence[Mapping[str, Any]],
) -> None:
    """Recompute significant topology and the declared multi-metric gate.

    Peak detection already requires one-percent relative prominence.  A raw
    missing/spurious row is additionally considered significant only when its
    reference/candidate basin carries at least one percent of the analysis
    area.  Raw topology counts remain untouched in the summary.
    """

    topology: Dict[Tuple[str, str], List[int]] = {}
    for item in sidebands:
        case_id = str(item.get("case_id", ""))
        approximation = str(item.get("approximation", "")).lower()
        if not case_id or approximation not in ("ba", "scba"):
            continue
        counts = topology.setdefault((case_id, approximation), [0, 0])
        status = str(item.get("status", ""))
        if status == "missing" and float(item.get("reference_weight", 0.0)) >= SIGNIFICANT_BASIN_WEIGHT:
            counts[0] += 1
        if status == "spurious" and float(item.get("candidate_weight", 0.0)) >= SIGNIFICANT_BASIN_WEIGHT:
            counts[1] += 1
    for case_id, row in summary.items():
        for approximation in ("ba", "scba"):
            missing, spurious = topology.get((case_id, approximation), [0, 0])
            row[approximation + "_significant_missing_count"] = missing
            row[approximation + "_significant_spurious_count"] = spurious
        row["significant_basin_weight_threshold"] = SIGNIFICANT_BASIN_WEIGHT
        l1 = float(row["scba_l1"])
        wasserstein = float(row["scba_w_over_omega0"])
        shift = float(row["tracked_scba_shift_over_omega0"])
        weight_error = float(row["tracked_scba_weight_error"])
        sideband_error = float(row["scba_sideband_weight_error"])
        area_error = float(row["scba_raw_area_relative_difference"])
        ed_ambiguity = str(row.get("tracked_ed_ambiguity", ""))
        scba_ambiguity = str(row.get("tracked_scba_ambiguity", ""))
        ed_weight = float(row["tracked_ed_basin_weight"])
        scba_weight = float(row["tracked_scba_basin_weight"])
        row["valid_l1"] = bool(l1 <= VALID_L1)
        row["valid_wasserstein"] = bool(wasserstein <= VALID_W_OVER_OMEGA)
        row["valid_tracked_shift"] = bool(np.isfinite(shift) and abs(shift) <= VALID_TRACKED_SHIFT_OVER_OMEGA)
        row["valid_tracked_weight"] = bool(np.isfinite(weight_error) and abs(weight_error) <= VALID_TRACKED_WEIGHT_ERROR)
        row["valid_sideband_weight"] = bool(np.isfinite(sideband_error) and sideband_error <= VALID_SIDEBAND_WEIGHT_ERROR)
        row["valid_peak_topology"] = bool(
            int(row["scba_significant_missing_count"]) == 0
            and int(row["scba_significant_spurious_count"]) == 0
        )
        row["valid_raw_area"] = bool(abs(area_error) <= VALID_RAW_AREA_RELATIVE_ERROR)
        row["valid_tracking"] = bool(
            not ed_ambiguity and not scba_ambiguity
            and ed_weight >= SIGNIFICANT_BASIN_WEIGHT
            and scba_weight >= SIGNIFICANT_BASIN_WEIGHT
        )
        row["operational_validity_pass"] = bool(all(row[key] for key in (
            "valid_l1", "valid_wasserstein", "valid_tracked_shift",
            "valid_tracked_weight", "valid_sideband_weight",
            "valid_peak_topology", "valid_raw_area", "valid_tracking",
        )))


def _join_baselines(summary: Dict[str, Dict[str, Any]]) -> None:
    primary_baselines = {
        round(float(row["huang_rhys"]), 12): row
        for row in summary.values() if row.get("family") == "primary_J0_baseline"
    }
    secondary_baselines = {
        round(float(row["r_abs"]), 12): row
        for row in summary.values() if row.get("family") == "secondary_J0_baseline"
    }
    for row in summary.values():
        family = row.get("family")
        baseline = None
        if family == "primary_fixed_S":
            baseline = primary_baselines.get(round(float(row["huang_rhys"]), 12))
        elif family == "secondary_fixed_lambda":
            baseline = secondary_baselines.get(round(float(row["r_abs"]), 12))
        if baseline is None:
            continue
        row["baseline_case_id"] = baseline["case_id"]
        row["baseline_scba_l1"] = baseline["scba_l1"]
        row["baseline_scba_w_over_omega0"] = baseline["scba_w_over_omega0"]
        row["resonance_excess_l1_diagnostic"] = float(row["scba_l1"]) - float(baseline["scba_l1"])
        row["resonance_excess_w_over_omega0_diagnostic"] = float(row["scba_w_over_omega0"]) - float(baseline["scba_w_over_omega0"])


def _poisson_green(energy: np.ndarray, model: ModelParams) -> np.ndarray:
    weights = [math.exp(-model.huang_rhys)]
    while 1.0 - sum(weights) > 1.0e-15:
        weights.append(weights[-1] * model.huang_rhys / len(weights))
        if len(weights) > 1000:
            raise RuntimeError("Poisson tail did not converge")
    result = np.zeros_like(energy, dtype=complex)
    for m, weight in enumerate(weights):
        center = model.energy_00_eV + m * model.omega0_eV
        result += weight / (energy - center + 1.0j * model.eta_eV)
    return result


def _grid_convergence_rows(
    cases: Sequence[FrequencyCase],
    primary_cutoffs: Mapping[float, int],
) -> List[Dict[str, Any]]:
    """Compare dx halving and dimensionless padding widening."""

    specifications = (
        ("weak_even_control", _model_for_r(0.02, 4.0, 1), primary_cutoffs[0.02]),
        ("weak_positive_r1", _model_for_r(0.02, 1.0, 1), primary_cutoffs[0.02]),
        ("weak_even_control_high_error", _model_for_r(0.05, 2.0, 1), primary_cutoffs[0.05]),
        ("strong_positive_r1", _model_for_r(2.0, 1.0, 1), primary_cutoffs[2.0]),
    )
    selected_secondary = next(
        case for case in cases
        if case.family == "secondary_fixed_lambda" and case.j_sign == 1
        and abs(case.r_abs - 5.0) < 1.0e-12
    )
    secondary_n, _ = determine_secondary_cutoff(selected_secondary)
    all_specs = specifications + (("low_omega_fixed_lambda", selected_secondary.model, secondary_n),)
    rows: List[Dict[str, Any]] = []
    for label, model, n_vib in all_specs:
        grids = {
            "base": dimensionless_grid(model),
            "dx_half": dimensionless_grid(model, dx=0.5 * DX),
            "padding_wide": dimensionless_grid(
                model, dx=DX, analysis_x=ANALYSIS_X, internal_x=(-40.0, 32.0)
            ),
        }
        solutions: Dict[str, Tuple[EDResult, ScbaResult]] = {}
        for variant, grid in grids.items():
            ed = solve_ed(model, grid, n_vib=n_vib, n_thermal=1,
                          thermal_tolerance=1.0e-12,
                          stick_weight_tolerance=1.0e-14,
                          deposition_refinement=4)
            scba = solve_scba(model, grid, ScbaNumerics(), seed=None,
                              use_zero_temperature_sweep=True, verbose=False)
            solutions[variant] = (ed, scba)
        base_ed, base_scba = solutions["base"]
        base_grid = grids["base"]
        window = (base_grid.analysis_min_eV, base_grid.analysis_max_eV)
        base_physical = compare_spectra(
            base_ed.energy_eV, base_ed.spectrum_bright,
            base_scba.spectrum_bright_per_eV, window=window,
        )
        for variant in ("dx_half", "padding_wide"):
            candidate_ed, candidate_scba = solutions[variant]
            target_energy = base_ed.energy_eV
            ed_interp = np.interp(target_energy, candidate_ed.energy_eV, candidate_ed.spectrum_bright)
            scba_interp = np.interp(
                target_energy, candidate_scba.omega_eV,
                candidate_scba.spectrum_bright_per_eV,
            )
            ed_change = compare_spectra(
                target_energy, base_ed.spectrum_bright, ed_interp, window=window
            )
            scba_change = compare_spectra(
                target_energy, base_scba.spectrum_bright_per_eV,
                scba_interp, window=window,
            )
            variant_physical = compare_spectra(
                candidate_ed.energy_eV, candidate_ed.spectrum_bright,
                candidate_scba.spectrum_bright_per_eV,
                window=(grids[variant].analysis_min_eV, grids[variant].analysis_max_eV),
            )
            ed_peaks = analyze_peaks(
                candidate_ed.energy_eV, candidate_ed.spectrum_bright,
                window=(grids[variant].analysis_min_eV, grids[variant].analysis_max_eV),
                prominence_fraction=PROMINENCE_FRACTION,
                minimum_distance_eV=MINIMUM_PEAK_DISTANCE_OVER_OMEGA * model.omega0_eV,
            )
            scba_peaks = analyze_peaks(
                candidate_scba.omega_eV, candidate_scba.spectrum_bright_per_eV,
                window=(grids[variant].analysis_min_eV, grids[variant].analysis_max_eV),
                prominence_fraction=PROMINENCE_FRACTION,
                minimum_distance_eV=MINIMUM_PEAK_DISTANCE_OVER_OMEGA * model.omega0_eV,
            )
            base_ed_peaks = analyze_peaks(
                base_ed.energy_eV, base_ed.spectrum_bright, window=window,
                prominence_fraction=PROMINENCE_FRACTION,
                minimum_distance_eV=MINIMUM_PEAK_DISTANCE_OVER_OMEGA * model.omega0_eV,
            )
            base_scba_peaks = analyze_peaks(
                base_scba.omega_eV, base_scba.spectrum_bright_per_eV, window=window,
                prominence_fraction=PROMINENCE_FRACTION,
                minimum_distance_eV=MINIMUM_PEAK_DISTANCE_OVER_OMEGA * model.omega0_eV,
            )
            rows.append({
                "case_label": label, "variant": variant,
                "huang_rhys": model.huang_rhys,
                "r_signed": model.resonance_ratio,
                "omega0_meV": model.omega0_meV, "n_vib": n_vib,
                "base_scba_vs_ed_l1": base_physical.unit_area_l1,
                "variant_scba_vs_ed_l1": variant_physical.unit_area_l1,
                "scba_vs_ed_l1_change": variant_physical.unit_area_l1 - base_physical.unit_area_l1,
                "base_scba_vs_ed_w_over_omega0": base_physical.wasserstein_eV / model.omega0_eV,
                "variant_scba_vs_ed_w_over_omega0": variant_physical.wasserstein_eV / model.omega0_eV,
                "scba_vs_ed_w_over_omega0_change": (
                    variant_physical.wasserstein_eV - base_physical.wasserstein_eV
                ) / model.omega0_eV,
                "ed_solution_l1_change": ed_change.unit_area_l1,
                "ed_solution_w_over_omega0_change": ed_change.wasserstein_eV / model.omega0_eV,
                "scba_solution_l1_change": scba_change.unit_area_l1,
                "scba_solution_w_over_omega0_change": scba_change.wasserstein_eV / model.omega0_eV,
                "base_ed_peak_count": len(base_ed_peaks.peaks),
                "variant_ed_peak_count": len(ed_peaks.peaks),
                "base_scba_peak_count": len(base_scba_peaks.peaks),
                "variant_scba_peak_count": len(scba_peaks.peaks),
            })
    return rows


def _j0_scaling_rows(primary_cutoffs: Mapping[float, int]) -> List[Dict[str, Any]]:
    """Verify that one cached J=0 baseline per S is dimensionlessly reusable."""

    def interpolate_complex(x_new: np.ndarray, x_old: np.ndarray,
                            values: np.ndarray) -> np.ndarray:
        return (np.interp(x_new, x_old, np.real(values))
                + 1.0j * np.interp(x_new, x_old, np.imag(values)))

    canonical_x = np.linspace(
        INTERNAL_X[0], INTERNAL_X[1],
        int(round((INTERNAL_X[1] - INTERNAL_X[0]) / DX)) + 1,
    )
    rows: List[Dict[str, Any]] = []
    for s_value in PRIMARY_S:
        solutions: Dict[float, Tuple[EDResult, ScbaResult, ModelParams]] = {}
        for omega0_meV in (10.0, 50.0, 200.0):
            model = ModelParams(
                energy_00_eV=2.0, omega0_meV=omega0_meV,
                huang_rhys=s_value, j_meV=0.0, temperature_K=0.0,
                eta_meV=ETA_OVER_OMEGA * omega0_meV,
            )
            grid = dimensionless_grid(model)
            ed = solve_ed(model, grid, n_vib=primary_cutoffs[s_value], n_thermal=1)
            scba = solve_scba(model, grid, ScbaNumerics())
            solutions[omega0_meV] = (ed, scba, model)
        reference_ed, reference_scba, reference_model = solutions[50.0]
        x_reference = (
            reference_ed.energy_eV - reference_model.vertical_energy_eV
        ) / reference_model.omega0_eV
        reference_ed_green = interpolate_complex(
            canonical_x, x_reference,
            reference_model.omega0_eV * reference_ed.green_bright,
        )
        reference_scba_green = interpolate_complex(
            canonical_x, x_reference,
            reference_model.omega0_eV * reference_scba.green_bright_per_eV,
        )
        for omega0_meV in (10.0, 200.0):
            ed, scba, model = solutions[omega0_meV]
            x = (ed.energy_eV - model.vertical_energy_eV) / model.omega0_eV
            nearest = np.clip(
                np.rint((canonical_x - x[0]) / DX).astype(int), 0, x.size - 1
            )
            candidate_ed_green = interpolate_complex(
                canonical_x, x, model.omega0_eV * ed.green_bright,
            )
            candidate_scba_green = interpolate_complex(
                canonical_x, x, model.omega0_eV * scba.green_bright_per_eV,
            )
            ed_spectrum_comparison = compare_spectra(
                canonical_x, -np.imag(reference_ed_green),
                -np.imag(candidate_ed_green), window=ANALYSIS_X,
            )
            scba_spectrum_comparison = compare_spectra(
                canonical_x, -np.imag(reference_scba_green),
                -np.imag(candidate_scba_green), window=ANALYSIS_X,
            )
            rows.append({
                "huang_rhys": s_value, "reference_omega0_meV": 50.0,
                "candidate_omega0_meV": omega0_meV,
                "candidate_grid_samples": int(x.size),
                "x_grid_max_abs": float(np.max(np.abs(x[nearest] - canonical_x))),
                "dimensionless_ed_green_max_abs": float(np.max(np.abs(
                    candidate_ed_green - reference_ed_green
                ))),
                "dimensionless_ed_green_max_relative": float(np.max(np.abs(
                    candidate_ed_green - reference_ed_green
                )) / max(float(np.max(np.abs(reference_ed_green))), 1.0e-300)),
                "dimensionless_ed_spectrum_l1": ed_spectrum_comparison.unit_area_l1,
                "dimensionless_ed_spectrum_w_over_omega0": ed_spectrum_comparison.wasserstein_eV,
                "dimensionless_scba_green_max_abs": float(np.max(np.abs(
                    candidate_scba_green - reference_scba_green
                ))),
                "dimensionless_scba_green_max_relative": float(np.max(np.abs(
                    candidate_scba_green - reference_scba_green
                )) / max(float(np.max(np.abs(reference_scba_green))), 1.0e-300)),
                "dimensionless_scba_spectrum_l1": scba_spectrum_comparison.unit_area_l1,
                "dimensionless_scba_spectrum_w_over_omega0": scba_spectrum_comparison.wasserstein_eV,
            })
    return rows


def run_validation(
    cases: Sequence[FrequencyCase],
    summary: Mapping[str, Mapping[str, Any]],
    cutoffs: Mapping[float, int],
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    def check(name: str, value: float, threshold: float, relation: str = "<=") -> None:
        passed = value <= threshold if relation == "<=" else value >= threshold
        checks.append({"name": name, "value": value, "threshold": threshold,
                       "relation": relation, "pass": bool(passed)})

    # Existing omega0=50 meV, r=+1 benchmark reproduction.  Window changes
    # only far Lorentzian tails, so use tight but non-bitwise metric tolerances.
    legacy_path = MODULE_DIR / "results" / "ed_scba_error_summary.csv"
    legacy_rows = _read_csv_rows(legacy_path)
    reproduction: List[Dict[str, Any]] = []
    for s_value in (0.05, 0.10, 0.25, 0.50, 1.0, 2.0):
        current_id = "primary_S%s_Jp_r1" % _tag(s_value)
        current = summary.get(current_id)
        candidates = [row for row in legacy_rows
                      if "heatmap_r_S" in row.get("manifest_families", "")
                      and abs(float(row["huang_rhys"]) - s_value) < 1.0e-12
                      and abs(float(row["r_2J_over_omega0"]) - 1.0) < 1.0e-12
                      and abs(float(row["temperature_K"])) < 1.0e-12]
        if current is None or len(candidates) != 1:
            continue
        old = candidates[0]
        l1_diff = abs(float(current["scba_l1"]) - float(old["scba_unit_area_l1"]))
        w_diff = abs(float(current["scba_w_over_omega0"]) - float(old["scba_wasserstein_eV"]) / 0.05)
        reproduction.append({"huang_rhys": s_value, "l1_abs_difference": l1_diff,
                             "w_over_omega_abs_difference": w_diff})
    if reproduction:
        check("legacy_r1_l1", max(row["l1_abs_difference"] for row in reproduction), 3.0e-4)
        check("legacy_r1_w_over_omega", max(row["w_over_omega_abs_difference"] for row in reproduction), 3.0e-4)

    # Exact dimensionless rescaling for ED and SCBA.
    scaling_rows: List[Dict[str, Any]] = []
    for s_value, r_abs, sign in ((0.05, 1.0, 1), (0.05, 2.0, -1), (0.50, 1.0, 1), (2.0, 3.0, 1)):
        base = _model_for_r(s_value, r_abs, sign)
        scaled = ModelParams(
            energy_00_eV=2.0, omega0_meV=2.0 * base.omega0_meV,
            huang_rhys=s_value, j_meV=2.0 * base.j_meV,
            temperature_K=0.0, eta_meV=2.0 * base.eta_meV,
        )
        n_vib = cutoffs[s_value]
        ed1 = solve_ed(base, dimensionless_grid(base), n_vib=n_vib, n_thermal=1)
        ed2 = solve_ed(scaled, dimensionless_grid(scaled), n_vib=n_vib, n_thermal=1)
        sc1 = solve_scba(base, dimensionless_grid(base), ScbaNumerics())
        sc2 = solve_scba(scaled, dimensionless_grid(scaled), ScbaNumerics())
        x1 = (ed1.energy_eV - (base.vertical_energy_eV + base.j_eV)) / base.omega0_eV
        x2 = (ed2.energy_eV - (scaled.vertical_energy_eV + scaled.j_eV)) / scaled.omega0_eV
        grid_error = float(np.max(np.abs(x1 - x2)))
        ed1_dimensionless = base.omega0_eV * ed1.green_bright
        ed2_dimensionless = scaled.omega0_eV * ed2.green_bright
        sc1_dimensionless = base.omega0_eV * sc1.green_bright_per_eV
        sc2_dimensionless = scaled.omega0_eV * sc2.green_bright_per_eV
        ed_error = float(np.max(np.abs(ed1_dimensionless - ed2_dimensionless)))
        scba_error = float(np.max(np.abs(sc1_dimensionless - sc2_dimensionless)))
        ed_spectrum_comparison = compare_spectra(
            x1, -np.imag(ed1_dimensionless), -np.imag(ed2_dimensionless),
            window=ANALYSIS_X,
        )
        scba_spectrum_comparison = compare_spectra(
            x1, -np.imag(sc1_dimensionless), -np.imag(sc2_dimensionless),
            window=ANALYSIS_X,
        )
        scaling_rows.append({"huang_rhys": s_value, "r_abs": r_abs, "j_sign": sign,
                             "x_grid_max_abs": grid_error,
                             "dimensionless_ed_green_max_abs": ed_error,
                             "dimensionless_ed_green_max_relative": ed_error / max(
                                 float(np.max(np.abs(ed1_dimensionless))), 1.0e-300),
                             "dimensionless_ed_spectrum_l1": ed_spectrum_comparison.unit_area_l1,
                             "dimensionless_ed_spectrum_w_over_omega0": ed_spectrum_comparison.wasserstein_eV,
                             "dimensionless_scba_green_max_abs": scba_error,
                             "dimensionless_scba_green_max_relative": scba_error / max(
                                 float(np.max(np.abs(sc1_dimensionless))), 1.0e-300),
                             "dimensionless_scba_spectrum_l1": scba_spectrum_comparison.unit_area_l1,
                             "dimensionless_scba_spectrum_w_over_omega0": scba_spectrum_comparison.wasserstein_eV})
    check("dimensionless_grid_scaling", max(row["x_grid_max_abs"] for row in scaling_rows), 2.0e-12)
    # ED diagonalizes a Hamiltonian retaining the 2 eV common offset; its
    # absolute pointwise scaling error is therefore limited by eigensolver
    # conditioning, while the relative error is the primary invariant check.
    check("dimensionless_ed_scaling_absolute", max(row["dimensionless_ed_green_max_abs"] for row in scaling_rows), 3.0e-6)
    check("dimensionless_ed_scaling_relative", max(row["dimensionless_ed_green_max_relative"] for row in scaling_rows), 1.0e-7)
    check("dimensionless_ed_spectrum_scaling_L1", max(row["dimensionless_ed_spectrum_l1"] for row in scaling_rows), 1.0e-5)
    check("dimensionless_ed_spectrum_scaling_W", max(row["dimensionless_ed_spectrum_w_over_omega0"] for row in scaling_rows), 1.0e-6)
    check("dimensionless_scba_scaling", max(row["dimensionless_scba_green_max_abs"] for row in scaling_rows), 2.0e-10)
    check("dimensionless_scba_scaling_relative", max(row["dimensionless_scba_green_max_relative"] for row in scaling_rows), 1.0e-7)
    check("dimensionless_scba_spectrum_scaling_L1", max(row["dimensionless_scba_spectrum_l1"] for row in scaling_rows), 1.0e-5)
    check("dimensionless_scba_spectrum_scaling_W", max(row["dimensionless_scba_spectrum_w_over_omega0"] for row in scaling_rows), 1.0e-6)

    # The seven production J=0 baselines are cached at omega0=50 meV.  Verify
    # directly that they are reusable at fixed S and eta/omega0 rather than
    # relying only on the finite-J scaling checks above.
    j0_scaling_rows = _j0_scaling_rows(cutoffs)
    check("J0_dimensionless_grid_scaling",
          max(row["x_grid_max_abs"] for row in j0_scaling_rows), 2.0e-12)
    check("J0_dimensionless_ed_scaling_absolute",
          max(row["dimensionless_ed_green_max_abs"] for row in j0_scaling_rows), 3.0e-6)
    check("J0_dimensionless_ed_scaling_relative",
          max(row["dimensionless_ed_green_max_relative"] for row in j0_scaling_rows), 1.0e-7)
    check("J0_dimensionless_ed_spectrum_scaling_L1",
          max(row["dimensionless_ed_spectrum_l1"] for row in j0_scaling_rows), 1.0e-5)
    check("J0_dimensionless_ed_spectrum_scaling_W",
          max(row["dimensionless_ed_spectrum_w_over_omega0"] for row in j0_scaling_rows), 1.0e-6)
    check("J0_dimensionless_scba_scaling",
          max(row["dimensionless_scba_green_max_abs"] for row in j0_scaling_rows), 2.0e-10)
    check("J0_dimensionless_scba_scaling_relative",
          max(row["dimensionless_scba_green_max_relative"] for row in j0_scaling_rows), 1.0e-7)
    check("J0_dimensionless_scba_spectrum_scaling_L1",
          max(row["dimensionless_scba_spectrum_l1"] for row in j0_scaling_rows), 1.0e-5)
    check("J0_dimensionless_scba_spectrum_scaling_W",
          max(row["dimensionless_scba_spectrum_w_over_omega0"] for row in j0_scaling_rows), 1.0e-6)

    poisson_rows: List[Dict[str, Any]] = []
    for s_value in (0.02, 0.50, 2.0):
        model = ModelParams(energy_00_eV=2.0, omega0_meV=50.0,
                            huang_rhys=s_value, j_meV=0.0,
                            temperature_K=0.0, eta_meV=0.5)
        grid = dimensionless_grid(model)
        ed = solve_ed(model, grid, n_vib=cutoffs[s_value], n_thermal=1)
        exact = _poisson_green(ed.energy_eV, model)
        comparison = compare_spectra(ed.energy_eV, -np.imag(exact), ed.spectrum_bright,
                                     (grid.analysis_min_eV, grid.analysis_max_eV))
        poisson_rows.append({"huang_rhys": s_value, "l1": comparison.unit_area_l1,
                             "w_over_omega0": comparison.wasserstein_eV / model.omega0_eV})
    check("J0_Poisson_L1", max(row["l1"] for row in poisson_rows), 2.0e-4)
    check("J0_Poisson_W", max(row["w_over_omega0"] for row in poisson_rows), 2.0e-5)

    parity_rows: List[Dict[str, Any]] = []
    for s_value in (0.10, 0.50, 1.0):
        n_vib = cutoffs[s_value]
        model = _model_for_r(s_value, 2.0, 1)
        number = np.diag(np.arange(n_vib, dtype=float))
        raising = np.diag(np.sqrt(np.arange(1, n_vib, dtype=float)), 1)
        coordinate = raising + raising.T
        tau_z = np.diag([1.0, -1.0]); tau_x = np.array([[0.0, 1.0], [1.0, 0.0]])
        h = (np.kron(model.vertical_energy_eV * np.eye(2) + model.j_eV * tau_z, np.eye(n_vib))
             + np.kron(np.eye(2), model.omega0_eV * number)
             + model.coupling_eV / math.sqrt(2.0) * np.kron(tau_x, coordinate))
        parity = np.diag(_parity_diagonal(n_vib))
        commutator = float(np.max(np.abs(h @ parity - parity @ h)) / model.omega0_eV)
        opposite = _parity_diagonal(n_vib)[:, None] != _parity_diagonal(n_vib)[None, :]
        forbidden = float(np.max(np.abs(h[opposite])) / model.omega0_eV)
        values, vectors = difference_mode_eigensystem(model, n_vib)
        parity_expectations = np.sum(_parity_diagonal(n_vib)[:, None] * np.square(np.abs(vectors)), axis=0)
        leakage = float(np.max(np.minimum(abs(parity_expectations - 1.0), abs(parity_expectations + 1.0))))
        parity_rows.append({"huang_rhys": s_value, "commutator_over_omega0": commutator,
                            "opposite_parity_hamiltonian_block_over_omega0": forbidden,
                            "eigenstate_parity_leakage": leakage,
                            "forbidden_even_avoided_gap_tolerance_over_omega0": 2.0e-12})
    check("parity_commutator", max(row["commutator_over_omega0"] for row in parity_rows), 2.0e-13)
    check("parity_forbidden_block", max(row["opposite_parity_hamiltonian_block_over_omega0"] for row in parity_rows), 2.0e-13)
    check("parity_eigenstate_leakage", max(row["eigenstate_parity_leakage"] for row in parity_rows), 2.0e-10)

    # Representative convergence in the actual dimensionless production
    # discretization, including the low-frequency fixed-lambda endpoint.
    grid_rows = _grid_convergence_rows(cases, cutoffs)
    dx_rows = [row for row in grid_rows if row["variant"] == "dx_half"]
    padding_rows = [row for row in grid_rows if row["variant"] == "padding_wide"]
    check("grid_halving_ED_solution_L1",
          max(row["ed_solution_l1_change"] for row in dx_rows), 5.0e-3)
    check("grid_halving_SCBA_solution_L1",
          max(row["scba_solution_l1_change"] for row in dx_rows), 5.0e-3)
    check("grid_halving_SCBA_vs_ED_L1_change",
          max(abs(row["scba_vs_ed_l1_change"]) for row in dx_rows), 5.0e-3)
    check("grid_halving_SCBA_vs_ED_W_change",
          max(abs(row["scba_vs_ed_w_over_omega0_change"]) for row in dx_rows), 5.0e-4)
    check("padding_ED_solution_L1",
          max(row["ed_solution_l1_change"] for row in padding_rows), 5.0e-5)
    check("padding_SCBA_solution_L1",
          max(row["scba_solution_l1_change"] for row in padding_rows), 5.0e-5)
    check("padding_SCBA_vs_ED_L1_change",
          max(abs(row["scba_vs_ed_l1_change"]) for row in padding_rows), 5.0e-5)
    check("padding_SCBA_vs_ED_W_change",
          max(abs(row["scba_vs_ed_w_over_omega0_change"]) for row in padding_rows), 5.0e-6)

    completed = [row for row in summary.values() if row.get("family") in (
        "primary_fixed_S", "primary_J0_baseline", "secondary_fixed_lambda", "secondary_J0_baseline")]
    if completed:
        check("all_case_raw_area", max(abs(float(row["scba_raw_area_relative_difference"])) for row in completed), VALID_RAW_AREA_RELATIVE_ERROR)
        check("all_case_ED_pole_weight", max(abs(float(row["ed_pole_weight_error"])) for row in completed), 3.0e-12)
        check("all_case_ED_first_moment", max(abs(float(row["ed_first_moment_error_over_omega0"])) for row in completed), 3.0e-10)
        check("all_case_SCBA_residual", max(abs(float(row["scba_fixed_point_abs_residual_over_omega0"])) for row in completed), 2.0e-12)
        check("all_case_SCBA_causality", max(float(row["scba_max_positive_im_sigma_over_omega0"]) for row in completed), 2.0e-12)
        check("all_case_channel_closure", max(float(row["scba_channel_closure_over_omega0"]) for row in completed), 2.0e-12)
        check("all_case_ED_spectral_positivity",
              min(float(row["ed_minimum_spectrum"]) for row in completed),
              -1.0e-12, relation=">=")
        check("all_case_SCBA_spectral_positivity",
              min(float(row["scba_minimum_spectrum"]) for row in completed),
              -1.0e-12, relation=">=")

    manifest_rows = _read_csv_rows(MANIFEST_PATH)
    expected_family_counts = {
        family: sum(case.family == family for case in cases)
        for family in ("primary_fixed_S", "primary_J0_baseline",
                       "secondary_fixed_lambda", "secondary_J0_baseline")
    }
    actual_family_counts = {
        family: sum(row.get("family") == family for row in manifest_rows)
        for family in expected_family_counts
    }
    manifest_diagnostics = {
        "expected_case_count": len(cases),
        "manifest_row_count": len(manifest_rows),
        "manifest_unique_case_count": len({row.get("case_id") for row in manifest_rows}),
        "manifest_complete_count": sum(row.get("status") == "complete" for row in manifest_rows),
        "summary_row_count": len(completed),
        "summary_unique_case_count": len({str(row.get("case_id")) for row in completed}),
        "expected_family_counts": expected_family_counts,
        "actual_family_counts": actual_family_counts,
        "expected_archive_count": sum(case.save_archive for case in cases),
        "missing_or_invalid_archives": [
            case.case_id for case in cases if case.save_archive and not _archive_complete(case)
        ],
    }
    check("manifest_expected_rows", abs(len(manifest_rows) - len(cases)), 0.0)
    check("manifest_unique_case_ids",
          abs(manifest_diagnostics["manifest_unique_case_count"] - len(cases)), 0.0)
    check("manifest_all_complete",
          abs(manifest_diagnostics["manifest_complete_count"] - len(cases)), 0.0)
    check("summary_expected_rows", abs(len(completed) - len(cases)), 0.0)
    check("summary_unique_case_ids",
          abs(manifest_diagnostics["summary_unique_case_count"] - len(cases)), 0.0)
    check("manifest_family_counts",
          sum(abs(actual_family_counts[key] - expected_family_counts[key])
              for key in expected_family_counts), 0.0)
    check("representative_archives_complete",
          len(manifest_diagnostics["missing_or_invalid_archives"]), 0.0)

    # Representative full 2x2 asymmetric-seed exchange regression.
    exchange: Dict[str, Any] = {}
    model = _model_for_r(0.10, 1.0, 1)
    grid = dimensionless_grid(model, dx=0.004, analysis_x=(-10.0, 12.0), internal_x=(-20.0, 18.0))
    reduced = solve_scba(model, grid, ScbaNumerics())
    full = solve_full_matrix_regression(model, reduced, ScbaNumerics(), seed_asymmetry=1.0e-3)
    exchange = full.diagnostic_dict()
    check("full_matrix_bright", full.relative_bright_green_difference, 2.0e-7)
    check("full_matrix_site_symmetry", full.relative_site_sigma_difference, 2.0e-7)

    validation = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "dimensionless_grid": {"analysis_x": ANALYSIS_X, "internal_x": INTERNAL_X, "dx": DX},
        "accuracy_gate": {
            "L1_max": VALID_L1, "W_over_omega0_max": VALID_W_OVER_OMEGA,
            "tracked_shift_over_omega0_max": VALID_TRACKED_SHIFT_OVER_OMEGA,
            "tracked_basin_weight_error_max": VALID_TRACKED_WEIGHT_ERROR,
            "aggregate_sideband_weight_error_max": VALID_SIDEBAND_WEIGHT_ERROR,
            "significant_missing_spurious_required": 0,
            "peak_prominence_fraction": PROMINENCE_FRACTION,
            "significant_basin_weight_fraction": SIGNIFICANT_BASIN_WEIGHT,
            "raw_area_relative_error_max": VALID_RAW_AREA_RELATIVE_ERROR,
            "note": "operational at eta/omega0=0.01, not a phase boundary",
        },
        "checks": checks, "pass": bool(all(item["pass"] for item in checks)),
        "legacy_reproduction": reproduction, "dimensionless_scaling": scaling_rows,
        "J0_dimensionless_scaling": j0_scaling_rows,
        "grid_padding_convergence": grid_rows,
        "J0_poisson": poisson_rows, "parity": parity_rows,
        "full_matrix_regression": exchange,
        "manifest": manifest_diagnostics,
    }
    VALIDATION_JSON.write_text(json.dumps(_jsonable(validation), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["frequency scan numerical validation", "generated_utc={}".format(validation["generated_utc"])]
    for item in checks:
        lines.append("{} {} value={:.8e} threshold {} {:.8e}".format(
            "PASS" if item["pass"] else "FAIL", item["name"], item["value"],
            item["relation"], item["threshold"]))
    lines.append("overall={}".format("PASS" if validation["pass"] else "FAIL"))
    VALIDATION_TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv_atomic(VALIDATION_DIR / "legacy_reproduction.csv", reproduction,
                      ("huang_rhys", "l1_abs_difference", "w_over_omega_abs_difference"))
    _write_csv_atomic(VALIDATION_DIR / "dimensionless_scaling.csv", scaling_rows,
                      ("huang_rhys", "r_abs", "j_sign", "x_grid_max_abs",
                       "dimensionless_ed_green_max_abs",
                       "dimensionless_ed_green_max_relative",
                       "dimensionless_ed_spectrum_l1",
                       "dimensionless_ed_spectrum_w_over_omega0",
                       "dimensionless_scba_green_max_abs",
                       "dimensionless_scba_green_max_relative",
                       "dimensionless_scba_spectrum_l1",
                       "dimensionless_scba_spectrum_w_over_omega0"))
    _write_csv_atomic(J0_SCALING_PATH, j0_scaling_rows,
                      ("huang_rhys", "reference_omega0_meV", "candidate_omega0_meV",
                       "candidate_grid_samples", "x_grid_max_abs",
                       "dimensionless_ed_green_max_abs",
                       "dimensionless_ed_green_max_relative",
                       "dimensionless_ed_spectrum_l1",
                       "dimensionless_ed_spectrum_w_over_omega0",
                       "dimensionless_scba_green_max_abs",
                       "dimensionless_scba_green_max_relative",
                       "dimensionless_scba_spectrum_l1",
                       "dimensionless_scba_spectrum_w_over_omega0"))
    _write_csv_atomic(GRID_CONVERGENCE_PATH, grid_rows,
                      ("case_label", "variant", "huang_rhys", "r_signed",
                       "omega0_meV", "n_vib", "base_scba_vs_ed_l1",
                       "variant_scba_vs_ed_l1", "scba_vs_ed_l1_change",
                       "base_scba_vs_ed_w_over_omega0",
                       "variant_scba_vs_ed_w_over_omega0",
                       "scba_vs_ed_w_over_omega0_change",
                       "ed_solution_l1_change", "ed_solution_w_over_omega0_change",
                       "scba_solution_l1_change", "scba_solution_w_over_omega0_change",
                       "base_ed_peak_count", "variant_ed_peak_count",
                       "base_scba_peak_count", "variant_scba_peak_count"))
    _write_csv_atomic(VALIDATION_DIR / "poisson_validation.csv", poisson_rows,
                      ("huang_rhys", "l1", "w_over_omega0"))
    _write_csv_atomic(VALIDATION_DIR / "parity_validation.csv", parity_rows,
                      ("huang_rhys", "commutator_over_omega0",
                       "opposite_parity_hamiltonian_block_over_omega0",
                       "eigenstate_parity_leakage",
                       "forbidden_even_avoided_gap_tolerance_over_omega0"))
    return validation


def _select_cases(cases: Sequence[FrequencyCase], stage: str,
                  case_id: Optional[str], limit: Optional[int]) -> List[FrequencyCase]:
    if stage == "primary":
        selected = [case for case in cases if case.family in ("primary_fixed_S", "primary_J0_baseline")]
    elif stage == "secondary":
        selected = [case for case in cases if case.family in ("secondary_fixed_lambda", "secondary_J0_baseline")]
    elif stage == "production":
        selected = list(cases)
    elif stage == "smoke":
        wanted = {
            "primary_S0p02_Jp_r1", "primary_S0p02_Jp_r2",
            "primary_S0p5_Jp_r1", "primary_S0p5_Jm_r1",
            "baseline_S0p02_w50", "baseline_S0p5_w50",
        }
        selected = [case for case in cases if case.case_id in wanted]
    else:
        selected = []
    if case_id is not None:
        selected = [case for case in selected if case.case_id == case_id]
        if not selected:
            raise ValueError("Unknown/inactive --case-id {}".format(case_id))
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "primary", "secondary", "levels", "validation", "production"), default="smoke")
    parser.add_argument("--overwrite", action="store_true")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--case-id")
    selector.add_argument("--limit", type=int)
    parser.add_argument("--checkpoint", type=int, default=10,
                        help="atomically flush tables after this many completed cases")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.checkpoint < 1:
        raise ValueError("--checkpoint must be positive")
    _ensure_dirs()
    cases = build_cases()
    primary_cutoffs, cutoff_rows = determine_primary_cutoffs()
    secondary_cutoffs: Dict[str, int] = {}
    for case in cases:
        if case.family.startswith("secondary_"):
            cutoff, rows = determine_secondary_cutoff(case)
            secondary_cutoffs[case.case_id] = cutoff
            cutoff_rows.extend(rows)
    _write_csv_atomic(CUTOFF_PATH, cutoff_rows,
                      ("family", "case_id", "huang_rhys", "r_abs", "n_low", "n_high",
                       "stick_wasserstein_over_omega0", "maximum_weight_missing",
                       "first_moment_error_over_omega0", "pass"))

    track_map = build_track_map(cases, primary_cutoffs)
    if args.stage in ("levels", "production"):
        missing_level_products = [
            path for path in (AUTHORITATIVE_LEVEL_PATH, AUTHORITATIVE_GAP_PATH)
            if not path.is_file()
        ]
        if missing_level_products:
            raise RuntimeError(
                "Authoritative parity products are missing; run exact_level_map.py: {}"
                .format(", ".join(str(path) for path in missing_level_products))
            )
        print("Using authoritative level products: {} ; {}".format(
            AUTHORITATIVE_LEVEL_PATH, AUTHORITATIVE_GAP_PATH))
    if args.stage == "levels":
        return 0

    old_manifest = _read_csv_by_id(MANIFEST_PATH)
    old_summary = _read_csv_by_id(SUMMARY_PATH)
    sideband_rows = _read_csv_rows(SIDEBAND_PATH)
    manifest_rows: List[Dict[str, Any]] = []
    case_by_id = {case.case_id: case for case in cases}
    for case in cases:
        n_vib = (primary_cutoffs[case.model.huang_rhys]
                 if case.family.startswith("primary_") else secondary_cutoffs[case.case_id])
        base = _manifest_base(case, n_vib)
        old = old_manifest.get(case.case_id)
        if old and old.get("definition_key") == base["definition_key"]:
            for key in ("status", "started_utc", "finished_utc", "runtime_s",
                        "error_type", "error_message", "error_traceback"):
                base[key] = old.get(key, base[key])
        manifest_rows.append(base)
    manifest_by_id = {row["case_id"]: row for row in manifest_rows}
    _write_csv_atomic(MANIFEST_PATH, manifest_rows, MANIFEST_COLUMNS)

    active = _select_cases(cases, args.stage, args.case_id, args.limit)
    if args.stage == "validation":
        _refresh_operational_validity(old_summary, sideband_rows)
        _join_baselines(old_summary)
        _write_csv_atomic(
            SUMMARY_PATH,
            [old_summary[case.case_id] for case in cases if case.case_id in old_summary],
            SUMMARY_COLUMNS,
        )
        _write_csv_atomic(
            BASELINE_PATH,
            [row for row in old_summary.values()
             if row.get("family") in ("primary_J0_baseline", "secondary_J0_baseline")],
            SUMMARY_COLUMNS,
        )
        validation = run_validation(cases, old_summary, primary_cutoffs)
        print("Validation {}: {}".format("PASS" if validation["pass"] else "FAIL", VALIDATION_JSON))
        return 0 if validation["pass"] else 1
    print("Frequency scan stage={} active={} total={} r_points={}".format(
        args.stage, len(active), len(cases), len(primary_r_grid())))

    summary: Dict[str, Dict[str, Any]] = {key: dict(value) for key, value in old_summary.items()}
    new_sidebands = list(sideband_rows)
    completed = skipped = failures = since_checkpoint = 0

    def checkpoint() -> None:
        _refresh_operational_validity(summary, new_sidebands)
        _join_baselines(summary)
        ordered_summary = [summary[case.case_id] for case in cases if case.case_id in summary]
        _write_csv_atomic(MANIFEST_PATH, manifest_rows, MANIFEST_COLUMNS)
        _write_csv_atomic(SUMMARY_PATH, ordered_summary, SUMMARY_COLUMNS)
        _write_csv_atomic(SIDEBAND_PATH, new_sidebands, SIDEBAND_COLUMNS)

    for ordinal, case in enumerate(active, start=1):
        manifest = manifest_by_id[case.case_id]
        n_vib = int(manifest["n_vib"])
        existing = summary.get(case.case_id)
        complete = (
            not args.overwrite and manifest.get("status") == "complete"
            and existing is not None and existing.get("schema_version") == SCHEMA_VERSION
            and _archive_complete(case)
        )
        if complete:
            skipped += 1
            print("[{}/{}] SKIP {}".format(ordinal, len(active), case.case_id))
            continue
        manifest.update({"status": "running", "started_utc": _utc_now(),
                         "finished_utc": "", "error_type": "", "error_message": "",
                         "error_traceback": ""})
        print("[{}/{}] RUN {} N={}".format(ordinal, len(active), case.case_id, n_vib))
        start = time.perf_counter()
        try:
            if case.case_id in track_map:
                track = track_map[case.case_id]
            else:
                track = _baseline_track(case, n_vib)
            row, matches = run_case(case, n_vib, track)
            summary[case.case_id] = row
            new_sidebands[:] = [item for item in new_sidebands if item.get("case_id") != case.case_id]
            new_sidebands.extend(matches)
            manifest.update({"status": "complete", "finished_utc": _utc_now(),
                             "runtime_s": time.perf_counter() - start,
                             "error_type": "", "error_message": "", "error_traceback": ""})
            completed += 1
            print("[{}/{}] DONE {} L1={:.4f} W/w={:.4f}".format(
                ordinal, len(active), case.case_id, float(row["scba_l1"]),
                float(row["scba_w_over_omega0"])))
        except Exception as error:
            failures += 1
            manifest.update({"status": "failed", "finished_utc": _utc_now(),
                             "runtime_s": time.perf_counter() - start,
                             "error_type": type(error).__name__,
                             "error_message": str(error).replace("\n", " "),
                             "error_traceback": traceback.format_exc().replace("\n", "\\n")})
            print("[{}/{}] FAILED {}: {}: {}".format(
                ordinal, len(active), case.case_id, type(error).__name__, error), file=sys.stderr)
        since_checkpoint += 1
        if since_checkpoint >= args.checkpoint:
            checkpoint(); since_checkpoint = 0
    checkpoint()

    baseline_rows = [row for row in summary.values()
                     if row.get("family") in ("primary_J0_baseline", "secondary_J0_baseline")]
    _write_csv_atomic(BASELINE_PATH, baseline_rows, SUMMARY_COLUMNS)

    validation_result = None
    if args.stage == "production" and failures == 0:
        validation_result = run_validation(cases, summary, primary_cutoffs)
    print("Scan summary completed={} skipped={} failed={} manifest={}".format(
        completed, skipped, failures, MANIFEST_PATH))
    if validation_result is not None:
        print("Validation {}: {}".format(
            "PASS" if validation_result["pass"] else "FAIL", VALIDATION_JSON))
    return 1 if failures or (validation_result is not None and not validation_result["pass"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
