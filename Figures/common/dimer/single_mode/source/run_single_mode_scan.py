#!/usr/bin/env python3
"""Deterministic ED/BA/SCBA scan driver for the single-mode dimer.

The production case list is defined entirely in this file and deduplicated by
its physical parameters.  If one case belongs to several requested scans, its
manifest ``families`` field retains every membership separated by semicolons.
Results are written case by case with atomic replacement, and the CSV manifest
is likewise replaced atomically after every state transition.  Interrupted
runs can therefore be resumed safely without mixing partial archives into the
analysis.

Run from this directory or from the repository root, for example::

    python3 run_single_mode_scan.py --stage smoke
    python3 run_single_mode_scan.py --stage direct
    python3 run_single_mode_scan.py --stage production --limit 10
    python3 run_single_mode_scan.py --stage production --case-id CASE_ID

The smoke stage has its own manifest and spectra below ``validation/``.  The
direct and production stages share the complete production manifest below
``results/``; ``direct`` merely selects the nine direct-analogue cases from
that complete manifest.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from calculate_single_mode_ed import EDResult, solve_ed  # noqa: E402
from calculate_single_mode_scba import (  # noqa: E402
    HISTORY_COLUMNS,
    FullMatrixRegression,
    ScbaNumerics,
    ScbaResult,
    solve_ba,
    solve_full_matrix_regression,
    solve_scba,
)
from single_mode_common import (  # noqa: E402
    GridSpec,
    ModelParams,
    RESULTS,
    ROOT,
    SPECTRA,
    VALIDATION,
    default_grid_spec,
    encode_metadata,
    ensure_directories,
    model_dict,
    recommended_n_vib,
    thermal_cutoff,
)


THERMAL_TOLERANCE = 1.0e-10
STICK_WEIGHT_TOLERANCE = 1.0e-14
DEPOSITION_REFINEMENT = 4
DIRECT_J_MEV = (-100.0, -50.0, -25.0, -12.4, 0.0, 12.4, 25.0, 50.0, 100.0)
HEATMAP_S = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0)
RESONANCE_R = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class ScanCase:
    case_id: str
    model: ModelParams
    families: Tuple[str, ...]
    run_full_matrix_regression: bool

    @property
    def family_text(self) -> str:
        return ";".join(self.families)


MANIFEST_COLUMNS = (
    "case_id",
    "families",
    "status",
    "output_file",
    "energy_00_eV",
    "omega0_meV",
    "huang_rhys",
    "lambda_meV",
    "j_meV",
    "r_2J_over_omega0",
    "temperature_K",
    "eta_meV",
    "analysis_min_eV",
    "analysis_max_eV",
    "internal_min_eV",
    "internal_max_eV",
    "step_meV",
    "grid_samples",
    "full_matrix_requested",
    "n_vib",
    "n_thermal",
    "thermal_omitted_probability",
    "ed_raw_bright_pole_weight",
    "ed_raw_dark_pole_weight",
    "ed_common_final_omitted_weight",
    "ed_bright_first_moment_error_eV",
    "ed_dark_first_moment_error_eV",
    "ed_minimum_bright_spectrum",
    "ed_minimum_dark_spectrum",
    "ed_analysis_area_bright",
    "ba_analysis_area_bright",
    "scba_analysis_area_bright",
    "ba_fixed_point_abs_residual_eV",
    "ba_fixed_point_relative_residual",
    "scba_iterations",
    "scba_fixed_point_abs_residual_eV",
    "scba_fixed_point_relative_residual",
    "scba_green_relative_change",
    "scba_final_mixing",
    "scba_max_positive_im_sigma_eV",
    "scba_minimum_bright_spectrum",
    "scba_minimum_dark_spectrum",
    "full_iterations",
    "full_relative_site_sigma_difference",
    "full_relative_mean_sigma_difference",
    "full_relative_bright_green_difference",
    "full_relative_dark_green_difference",
    "full_symmetric_map_identity_error_eV",
    "full_symmetric_green_identity_error_per_eV",
    "started_utc",
    "finished_utc",
    "runtime_ed_s",
    "runtime_ba_s",
    "runtime_scba_s",
    "runtime_full_matrix_s",
    "runtime_save_s",
    "runtime_total_s",
    "diagnostics_json",
    "error_type",
    "error_message",
    "error_traceback",
)


REQUIRED_ARCHIVE_KEYS = (
    "energy_eV",
    "analysis_mask",
    "ed_green_bright_per_eV",
    "ed_green_dark_per_eV",
    "ed_green_local_per_eV",
    "ba_green_bright_per_eV",
    "ba_green_dark_per_eV",
    "ba_green_local_per_eV",
    "scba_green_bright_per_eV",
    "scba_green_dark_per_eV",
    "scba_green_local_per_eV",
    "scba_sigma_eV",
    "scba_sigma_from_bright_emission_eV",
    "scba_sigma_from_bright_absorption_eV",
    "scba_sigma_from_dark_emission_eV",
    "scba_sigma_from_dark_absorption_eV",
    "ba_history",
    "scba_history",
    "full_history",
    "metadata_json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_float(value: float) -> str:
    if abs(value) < 5.0e-15:
        value = 0.0
    return format(float(value), ".12g")


def _number_tag(value: float) -> str:
    text = _canonical_float(value)
    return text.replace("-", "m").replace("+", "p").replace(".", "p")


def _physical_key(model: ModelParams) -> Tuple[str, ...]:
    return (
        _canonical_float(model.energy_00_eV),
        _canonical_float(model.omega0_meV),
        _canonical_float(model.huang_rhys),
        _canonical_float(model.j_meV),
        _canonical_float(model.temperature_K),
        _canonical_float(model.eta_meV),
    )


def _case_id(model: ModelParams) -> str:
    return (
        "w%s_S%s_J%s_T%s_eta%s"
        % (
            _number_tag(model.omega0_meV),
            _number_tag(model.huang_rhys),
            _number_tag(model.j_meV),
            _number_tag(model.temperature_K),
            _number_tag(model.eta_meV),
        )
    )


def _model(
    omega0_meV: float,
    huang_rhys: float,
    j_meV: float,
    temperature_K: float,
    eta_meV: float,
) -> ModelParams:
    result = ModelParams(
        energy_00_eV=2.0,
        omega0_meV=float(omega0_meV),
        huang_rhys=float(huang_rhys),
        j_meV=float(j_meV),
        temperature_K=float(temperature_K),
        eta_meV=float(eta_meV),
    )
    result.validate()
    return result


def _add_case(
    ordered: Dict[Tuple[str, ...], Dict[str, Any]],
    model: ModelParams,
    family: str,
    full_matrix: bool = False,
) -> None:
    key = _physical_key(model)
    if key not in ordered:
        ordered[key] = {
            "model": model,
            "families": [family],
            "full_matrix": bool(full_matrix),
        }
        return
    entry = ordered[key]
    if family not in entry["families"]:
        entry["families"].append(family)
    entry["full_matrix"] = bool(entry["full_matrix"] or full_matrix)


def build_production_cases() -> List[ScanCase]:
    """Return the complete deduplicated production manifest in fixed order."""

    ordered: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    # A. Direct analogue of the Drude-dimer benchmark.
    direct_s = 30.0 / 24.8
    for j_meV in DIRECT_J_MEV:
        _add_case(
            ordered,
            _model(24.8, direct_s, j_meV, 300.0, 0.5),
            "direct_analogue",
            full_matrix=j_meV in (-12.4, 0.0, 12.4),
        )

    # B/C. Dimensionless resonance and coupling-strength heat map.
    for temperature_K in (0.0, 300.0):
        for huang_rhys in HEATMAP_S:
            for resonance_ratio in RESONANCE_R:
                _add_case(
                    ordered,
                    _model(
                        50.0,
                        huang_rhys,
                        0.5 * resonance_ratio * 50.0,
                        temperature_K,
                        0.5,
                    ),
                    "heatmap_r_S",
                )

    # Large-splitting extension used to test approach to the surviving
    # common-mode (S/2) Franck--Condon limit.  The r=+/-4 points merge with
    # the heat map; r=+/-8,+/-16 extend the asymptotic lever arm.
    for huang_rhys in (0.10, 1.0, 2.0):
        for resonance_ratio in (-16.0, -8.0, -4.0, 4.0, 8.0, 16.0):
            _add_case(
                ordered,
                _model(
                    50.0,
                    huang_rhys,
                    0.5 * resonance_ratio * 50.0,
                    0.0,
                    0.5,
                ),
                "large_splitting",
            )

    # D. Reduced frequency and temperature scan.
    for omega0_meV in (24.8, 50.0, 100.0, 200.0):
        for temperature_K in (0.0, 100.0, 300.0):
            for huang_rhys in (0.10, 1.0):
                for resonance_ratio in (-1.0, 1.0):
                    _add_case(
                        ordered,
                        _model(
                            omega0_meV,
                            huang_rhys,
                            0.5 * resonance_ratio * omega0_meV,
                            temperature_K,
                            0.5,
                        ),
                        "frequency_temperature",
                    )

    # E. Resolution scan in four qualitatively different regimes.
    eta_regimes = (
        (0.10, 4.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
    )
    for huang_rhys, resonance_ratio, temperature_K in eta_regimes:
        for eta_meV in (0.10, 0.25, 0.50, 1.0, 2.0, 5.0):
            _add_case(
                ordered,
                _model(
                    100.0,
                    huang_rhys,
                    0.5 * resonance_ratio * 100.0,
                    temperature_K,
                    eta_meV,
                ),
                "eta_regimes",
            )

    # Extra eta convergence at the direct physical parameters.  eta=0.5 meV
    # already belongs to the nine-case direct analogue and is not repeated.
    for j_meV in (-12.4, 0.0, 12.4):
        for eta_meV in (0.10, 0.25, 1.0, 2.0):
            _add_case(
                ordered,
                _model(24.8, direct_s, j_meV, 300.0, eta_meV),
                "direct_eta_convergence",
            )

    cases = [
        ScanCase(
            case_id=_case_id(entry["model"]),
            model=entry["model"],
            families=tuple(entry["families"]),
            run_full_matrix_regression=bool(entry["full_matrix"]),
        )
        for entry in ordered.values()
    ]
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("The deterministic case-ID scheme produced a collision")
    return cases


def build_smoke_cases() -> List[ScanCase]:
    """Return one T=0 and one finite-T inexpensive end-to-end case."""

    models = (
        _model(100.0, 0.10, 50.0, 0.0, 2.0),
        _model(100.0, 0.10, 50.0, 300.0, 2.0),
    )
    return [
        ScanCase(
            case_id="smoke_" + _case_id(model),
            model=model,
            families=("smoke",),
            run_full_matrix_regression=False,
        )
        for model in models
    ]


def _output_path(case: ScanCase, smoke: bool) -> Path:
    directory = VALIDATION / "smoke_spectra" if smoke else SPECTRA
    return directory / (case.case_id + ".npz")


def _manifest_path(smoke: bool) -> Path:
    return VALIDATION / "smoke_scan_manifest.csv" if smoke else RESULTS / "scan_manifest.csv"


def _empty_manifest_row(case: ScanCase, smoke: bool) -> Dict[str, str]:
    model = case.model
    grid_spec = default_grid_spec(model)
    grid_samples = int(grid_spec.grid().size)
    n_thermal, omitted = thermal_cutoff(model, THERMAL_TOLERANCE)
    n_vib = recommended_n_vib(model, n_thermal)
    output = _output_path(case, smoke)
    try:
        output_text = str(output.relative_to(ROOT))
    except ValueError:
        output_text = str(output)
    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "case_id": case.case_id,
            "families": case.family_text,
            "status": "pending",
            "output_file": output_text,
            "energy_00_eV": _canonical_float(model.energy_00_eV),
            "omega0_meV": _canonical_float(model.omega0_meV),
            "huang_rhys": _canonical_float(model.huang_rhys),
            "lambda_meV": _canonical_float(
                model.huang_rhys * model.omega0_meV
            ),
            "j_meV": _canonical_float(model.j_meV),
            "r_2J_over_omega0": _canonical_float(model.resonance_ratio),
            "temperature_K": _canonical_float(model.temperature_K),
            "eta_meV": _canonical_float(model.eta_meV),
            "analysis_min_eV": _canonical_float(grid_spec.analysis_min_eV),
            "analysis_max_eV": _canonical_float(grid_spec.analysis_max_eV),
            "internal_min_eV": _canonical_float(grid_spec.internal_min_eV),
            "internal_max_eV": _canonical_float(grid_spec.internal_max_eV),
            "step_meV": _canonical_float(grid_spec.step_meV),
            "grid_samples": str(grid_samples),
            "full_matrix_requested": str(case.run_full_matrix_regression).lower(),
            "n_vib": str(n_vib),
            "n_thermal": str(n_thermal),
            "thermal_omitted_probability": format(omitted, ".17g"),
        }
    )
    return row


def _load_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "case_id" not in reader.fieldnames:
            raise ValueError("Existing manifest has no case_id column: %s" % path)
        return {str(row["case_id"]): dict(row) for row in reader}


def _merge_previous_rows(
    cases: Sequence[ScanCase],
    smoke: bool,
    previous: Dict[str, Dict[str, str]],
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for case in cases:
        base = _empty_manifest_row(case, smoke)
        old = previous.get(case.case_id)
        if old is not None:
            # Physical/grid columns always come from the current deterministic
            # definition.  Only prior execution state and diagnostics survive.
            for column in MANIFEST_COLUMNS:
                if column in (
                    "case_id",
                    "families",
                    "output_file",
                    "energy_00_eV",
                    "omega0_meV",
                    "huang_rhys",
                    "lambda_meV",
                    "j_meV",
                    "r_2J_over_omega0",
                    "temperature_K",
                    "eta_meV",
                    "analysis_min_eV",
                    "analysis_max_eV",
                    "internal_min_eV",
                    "internal_max_eV",
                    "step_meV",
                    "grid_samples",
                    "full_matrix_requested",
                    "n_vib",
                    "n_thermal",
                    "thermal_omitted_probability",
                ):
                    continue
                if old.get(column, "") != "":
                    base[column] = old[column]
        rows.append(base)
    return rows


def _write_manifest_atomic(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            writer = csv.DictWriter(
                handle,
                fieldnames=MANIFEST_COLUMNS,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise


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


def _metadata(
    case: ScanCase,
    grid_spec: GridSpec,
    numerics: ScbaNumerics,
    ed: EDResult,
    ba: ScbaResult,
    scba: ScbaResult,
    full: Optional[FullMatrixRegression],
    runtimes: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "format": "single_mode_dimer_ed_ba_scba_v1",
        "case_id": case.case_id,
        "families": list(case.families),
        "model": model_dict(case.model),
        "grid": asdict(grid_spec),
        "ed_numerics": {
            "thermal_tolerance": THERMAL_TOLERANCE,
            "stick_weight_tolerance": STICK_WEIGHT_TOLERANCE,
            "deposition_refinement": DEPOSITION_REFINEMENT,
        },
        "scba_numerics": asdict(numerics),
        "ed": ed.diagnostic_dict(),
        "ba": ba.diagnostic_dict(),
        "scba": scba.diagnostic_dict(),
        "full_matrix_regression": (
            None if full is None else full.diagnostic_dict()
        ),
        "runtimes_seconds": runtimes,
        "history_columns": list(HISTORY_COLUMNS),
        "spectrum_convention": "A_X(E) = -Im G_X^R(E), raw full-line area pi",
        "phonon_linewidth": "none; eta is only the common retarded HWHM",
        "archive_precision": (
            "energy float64; Green functions and self-energies complex64; "
            "iteration histories float64"
        ),
        "archive_frequency_extent": (
            "analysis window plus one bracketing sample on each available side"
        ),
        "runtime_note": (
            "The manifest contains authoritative measured save and total times; "
            "archive metadata is assembled immediately before its single atomic write."
        ),
    }


def _save_combined_atomic(
    path: Path,
    case: ScanCase,
    grid_spec: GridSpec,
    numerics: ScbaNumerics,
    ed: EDResult,
    ba: ScbaResult,
    scba: ScbaResult,
    full: Optional[FullMatrixRegression],
    runtimes: Dict[str, float],
) -> None:
    full_energy = scba.omega_eV
    if not (
        np.array_equal(ed.energy_eV, full_energy)
        and np.array_equal(ba.omega_eV, full_energy)
    ):
        raise RuntimeError("ED, BA, and SCBA did not use the identical energy grid")
    full_analysis_mask = grid_spec.analysis_mask(full_energy)
    selected = np.flatnonzero(full_analysis_mask)
    if selected.size < 3:
        raise RuntimeError("Analysis window contains fewer than three samples")
    first = max(0, int(selected[0]) - 1)
    last = min(full_energy.size - 1, int(selected[-1]) + 1)
    saved = slice(first, last + 1)
    energy = full_energy[saved].astype(np.float64, copy=False)

    def compact(values: np.ndarray) -> np.ndarray:
        return np.asarray(values[saved], dtype=np.complex64)

    arrays: Dict[str, np.ndarray] = {
        "energy_eV": energy,
        "analysis_mask": full_analysis_mask[saved],
        "ed_green_bright_per_eV": compact(ed.green_bright_per_eV),
        "ed_green_dark_per_eV": compact(ed.green_dark_per_eV),
        "ed_green_local_per_eV": compact(ed.green_local_per_eV),
        "ba_green_bright_per_eV": compact(ba.green_bright_per_eV),
        "ba_green_dark_per_eV": compact(ba.green_dark_per_eV),
        "ba_green_local_per_eV": compact(ba.green_local_per_eV),
        "scba_green_bright_per_eV": compact(scba.green_bright_per_eV),
        "scba_green_dark_per_eV": compact(scba.green_dark_per_eV),
        "scba_green_local_per_eV": compact(scba.green_local_per_eV),
        "scba_sigma_eV": compact(scba.sigma_eV),
        "scba_sigma_from_bright_emission_eV": (
            compact(scba.sigma_from_bright_emission_eV)
        ),
        "scba_sigma_from_bright_absorption_eV": (
            compact(scba.sigma_from_bright_absorption_eV)
        ),
        "scba_sigma_from_dark_emission_eV": compact(
            scba.sigma_from_dark_emission_eV
        ),
        "scba_sigma_from_dark_absorption_eV": (
            compact(scba.sigma_from_dark_absorption_eV)
        ),
        "ba_history": ba.history,
        "scba_history": scba.history,
        "full_history": (
            np.empty((0, len(HISTORY_COLUMNS)), dtype=float)
            if full is None
            else full.history
        ),
    }
    arrays["metadata_json"] = encode_metadata(
        _jsonable(_metadata(case, grid_spec, numerics, ed, ba, scba, full, runtimes))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, str(path))
    except Exception:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
        raise


def _archive_is_complete(path: Path, case: ScanCase) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            if any(key not in archive.files for key in REQUIRED_ARCHIVE_KEYS):
                return False
            energy = np.asarray(archive["energy_eV"], dtype=float)
            analysis = np.asarray(archive["analysis_mask"], dtype=bool)
            if energy.ndim != 1 or energy.size < 3 or analysis.shape != energy.shape:
                return False
            for key in REQUIRED_ARCHIVE_KEYS:
                if key in ("metadata_json", "ba_history", "scba_history", "full_history"):
                    continue
                values = np.asarray(archive[key])
                if values.shape != energy.shape or not np.all(np.isfinite(values)):
                    return False
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
            if metadata.get("case_id") != case.case_id:
                return False
            if tuple(metadata.get("families", ())) != tuple(case.families):
                return False
            saved_model = metadata.get("model", {})
            expected_model = model_dict(case.model)
            for name in (
                "energy_00_eV",
                "omega0_meV",
                "huang_rhys",
                "j_meV",
                "temperature_K",
                "eta_meV",
            ):
                if name not in saved_model or not np.isclose(
                    float(saved_model[name]),
                    float(expected_model[name]),
                    rtol=0.0,
                    atol=1.0e-13,
                ):
                    return False
            saved_grid = metadata.get("grid", {})
            expected_grid = asdict(default_grid_spec(case.model))
            for name, expected_value in expected_grid.items():
                if name not in saved_grid or not np.isclose(
                    float(saved_grid[name]),
                    float(expected_value),
                    rtol=0.0,
                    atol=1.0e-13,
                ):
                    return False
            expected_analysis = default_grid_spec(case.model).analysis_mask(energy)
            if not np.array_equal(analysis, expected_analysis):
                return False
            saved_full = metadata.get("full_matrix_regression")
            if case.run_full_matrix_regression != isinstance(saved_full, dict):
                return False
            full_history = np.asarray(archive["full_history"])
            if case.run_full_matrix_regression and full_history.size == 0:
                return False
            if not case.run_full_matrix_regression and full_history.size != 0:
                return False
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def _analysis_area(spectrum: np.ndarray, energy: np.ndarray, mask: np.ndarray) -> float:
    return float(np.trapz(spectrum[mask], energy[mask]))


def _update_success_row(
    row: Dict[str, str],
    ed: EDResult,
    ba: ScbaResult,
    scba: ScbaResult,
    full: Optional[FullMatrixRegression],
    grid_spec: GridSpec,
    runtimes: Dict[str, float],
    metadata: Dict[str, Any],
) -> None:
    diagnostic = ed.diagnostics
    mask = grid_spec.analysis_mask(scba.omega_eV)
    row.update(
        {
            "status": "complete",
            "finished_utc": _utc_now(),
            "n_vib": str(diagnostic["n_vib"]),
            "n_thermal": str(diagnostic["n_thermal"]),
            "thermal_omitted_probability": format(
                diagnostic["thermal_omitted_probability"], ".17g"
            ),
            "ed_raw_bright_pole_weight": format(
                diagnostic["raw_bright_pole_weight"], ".17g"
            ),
            "ed_raw_dark_pole_weight": format(
                diagnostic["raw_dark_pole_weight"], ".17g"
            ),
            "ed_common_final_omitted_weight": format(
                diagnostic["common_final_basis_omitted_weight"], ".17g"
            ),
            "ed_bright_first_moment_error_eV": format(
                diagnostic["bright_first_moment_error_eV"], ".17g"
            ),
            "ed_dark_first_moment_error_eV": format(
                diagnostic["dark_first_moment_error_eV"], ".17g"
            ),
            "ed_minimum_bright_spectrum": format(
                diagnostic["minimum_bright_spectrum"], ".17g"
            ),
            "ed_minimum_dark_spectrum": format(
                diagnostic["minimum_dark_spectrum"], ".17g"
            ),
            "ed_analysis_area_bright": format(
                _analysis_area(ed.spectrum_bright, ed.energy_eV, mask), ".17g"
            ),
            "ba_analysis_area_bright": format(
                _analysis_area(ba.spectrum_bright_per_eV, ba.omega_eV, mask),
                ".17g",
            ),
            "scba_analysis_area_bright": format(
                _analysis_area(
                    scba.spectrum_bright_per_eV,
                    scba.omega_eV,
                    mask,
                ),
                ".17g",
            ),
            "ba_fixed_point_abs_residual_eV": format(
                ba.fixed_point_abs_residual_eV, ".17g"
            ),
            "ba_fixed_point_relative_residual": format(
                ba.fixed_point_relative_residual, ".17g"
            ),
            "scba_iterations": str(scba.iterations),
            "scba_fixed_point_abs_residual_eV": format(
                scba.fixed_point_abs_residual_eV, ".17g"
            ),
            "scba_fixed_point_relative_residual": format(
                scba.fixed_point_relative_residual, ".17g"
            ),
            "scba_green_relative_change": format(
                scba.green_relative_change, ".17g"
            ),
            "scba_final_mixing": format(scba.final_mixing, ".17g"),
            "scba_max_positive_im_sigma_eV": format(
                scba.max_positive_im_sigma_eV, ".17g"
            ),
            "scba_minimum_bright_spectrum": format(
                scba.minimum_bright_spectrum_per_eV, ".17g"
            ),
            "scba_minimum_dark_spectrum": format(
                scba.minimum_dark_spectrum_per_eV, ".17g"
            ),
            "runtime_ed_s": format(runtimes["ed"], ".9g"),
            "runtime_ba_s": format(runtimes["ba"], ".9g"),
            "runtime_scba_s": format(runtimes["scba"], ".9g"),
            "runtime_full_matrix_s": format(runtimes["full_matrix"], ".9g"),
            "runtime_save_s": format(runtimes["save"], ".9g"),
            "runtime_total_s": format(runtimes["total"], ".9g"),
            "diagnostics_json": json.dumps(
                _jsonable(metadata), sort_keys=True, separators=(",", ":")
            ),
            "error_type": "",
            "error_message": "",
            "error_traceback": "",
        }
    )
    full_columns = (
        "full_iterations",
        "full_relative_site_sigma_difference",
        "full_relative_mean_sigma_difference",
        "full_relative_bright_green_difference",
        "full_relative_dark_green_difference",
        "full_symmetric_map_identity_error_eV",
        "full_symmetric_green_identity_error_per_eV",
    )
    if full is None:
        for column in full_columns:
            row[column] = ""
    else:
        row.update(
            {
                "full_iterations": str(full.iterations),
                "full_relative_site_sigma_difference": format(
                    full.relative_site_sigma_difference, ".17g"
                ),
                "full_relative_mean_sigma_difference": format(
                    full.relative_mean_sigma_difference, ".17g"
                ),
                "full_relative_bright_green_difference": format(
                    full.relative_bright_green_difference, ".17g"
                ),
                "full_relative_dark_green_difference": format(
                    full.relative_dark_green_difference, ".17g"
                ),
                "full_symmetric_map_identity_error_eV": format(
                    full.symmetric_map_identity_max_error_eV, ".17g"
                ),
                "full_symmetric_green_identity_error_per_eV": format(
                    full.symmetric_green_identity_max_error_per_eV, ".17g"
                ),
            }
        )


def _run_case(
    case: ScanCase,
    row: Dict[str, str],
    numerics: ScbaNumerics,
    smoke: bool,
) -> None:
    model = case.model
    grid_spec = default_grid_spec(model)
    output = _output_path(case, smoke)
    n_thermal, _ = thermal_cutoff(model, THERMAL_TOLERANCE)
    n_vib = recommended_n_vib(model, n_thermal)
    total_start = time.perf_counter()

    step_start = time.perf_counter()
    ed = solve_ed(
        model,
        grid_spec,
        n_vib=n_vib,
        n_thermal=n_thermal,
        thermal_tolerance=THERMAL_TOLERANCE,
        stick_weight_tolerance=STICK_WEIGHT_TOLERANCE,
        deposition_refinement=DEPOSITION_REFINEMENT,
    )
    runtime_ed = time.perf_counter() - step_start

    step_start = time.perf_counter()
    ba = solve_ba(model, grid_spec, numerics)
    runtime_ba = time.perf_counter() - step_start

    step_start = time.perf_counter()
    scba = solve_scba(
        model,
        grid_spec,
        numerics,
        seed=None,
        use_zero_temperature_sweep=True,
        verbose=False,
    )
    runtime_scba = time.perf_counter() - step_start

    full: Optional[FullMatrixRegression] = None
    runtime_full = 0.0
    if case.run_full_matrix_regression:
        step_start = time.perf_counter()
        full = solve_full_matrix_regression(
            model,
            scba,
            numerics,
            seed_asymmetry=1.0e-3,
            verbose=False,
        )
        runtime_full = time.perf_counter() - step_start

    runtimes = {
        "ed": runtime_ed,
        "ba": runtime_ba,
        "scba": runtime_scba,
        "full_matrix": runtime_full,
        "save": 0.0,
        "total": time.perf_counter() - total_start,
    }
    save_start = time.perf_counter()
    _save_combined_atomic(
        output,
        case,
        grid_spec,
        numerics,
        ed,
        ba,
        scba,
        full,
        runtimes,
    )
    runtimes["save"] = time.perf_counter() - save_start
    runtimes["total"] = time.perf_counter() - total_start
    metadata = _metadata(case, grid_spec, numerics, ed, ba, scba, full, runtimes)
    _update_success_row(
        row,
        ed,
        ba,
        scba,
        full,
        grid_spec,
        runtimes,
        metadata,
    )


def _select_cases(
    all_cases: Sequence[ScanCase],
    stage: str,
    case_id: Optional[str],
    limit: Optional[int],
) -> List[ScanCase]:
    if stage == "direct":
        selected = [case for case in all_cases if "direct_analogue" in case.families]
    else:
        selected = list(all_cases)
    if case_id is not None:
        selected = [case for case in selected if case.case_id == case_id]
        if not selected:
            raise ValueError(
                "case ID %r does not belong to stage %s" % (case_id, stage)
            )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[:limit]
    return selected


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "direct", "production"),
        default="smoke",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute selected cases even when a complete archive exists",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--case-id", help="run one deterministic manifest case")
    selector.add_argument("--limit", type=int, help="run only the first N selected cases")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    smoke = args.stage == "smoke"
    all_cases = build_smoke_cases() if smoke else build_production_cases()
    active_cases = _select_cases(
        all_cases,
        args.stage,
        args.case_id,
        args.limit,
    )
    ensure_directories()
    manifest_path = _manifest_path(smoke)
    previous = _load_manifest(manifest_path)
    rows = _merge_previous_rows(all_cases, smoke, previous)
    rows_by_id = {row["case_id"]: row for row in rows}
    _write_manifest_atomic(manifest_path, rows)

    print(
        "Stage %s: %d active of %d manifest cases; manifest=%s"
        % (args.stage, len(active_cases), len(all_cases), manifest_path)
    )
    numerics = ScbaNumerics()
    failures = 0
    completed = 0
    skipped = 0

    for ordinal, case in enumerate(active_cases, start=1):
        row = rows_by_id[case.case_id]
        output = _output_path(case, smoke)
        if not args.overwrite and _archive_is_complete(output, case):
            row["status"] = "complete"
            row["error_type"] = ""
            row["error_message"] = ""
            row["error_traceback"] = ""
            _write_manifest_atomic(manifest_path, rows)
            skipped += 1
            print(
                "[%d/%d] SKIP complete %s"
                % (ordinal, len(active_cases), case.case_id)
            )
            continue

        row.update(
            {
                "status": "running",
                "started_utc": _utc_now(),
                "finished_utc": "",
                "error_type": "",
                "error_message": "",
                "error_traceback": "",
            }
        )
        _write_manifest_atomic(manifest_path, rows)
        print(
            "[%d/%d] RUN %s families=%s"
            % (ordinal, len(active_cases), case.case_id, case.family_text)
        )
        case_start = time.perf_counter()
        try:
            _run_case(case, row, numerics, smoke)
            completed += 1
            print(
                "[%d/%d] DONE %s total=%ss SCBA_iter=%s"
                % (
                    ordinal,
                    len(active_cases),
                    case.case_id,
                    row["runtime_total_s"],
                    row["scba_iterations"],
                )
            )
        except Exception as error:  # Keep independent cases running.
            failures += 1
            row.update(
                {
                    "status": "failed",
                    "finished_utc": _utc_now(),
                    "error_type": type(error).__name__,
                    "error_message": str(error).replace("\n", " "),
                    "error_traceback": traceback.format_exc().replace("\n", "\\n"),
                    "runtime_total_s": format(
                        time.perf_counter() - case_start,
                        ".9g",
                    ),
                }
            )
            print(
                "[%d/%d] FAILED %s: %s: %s"
                % (
                    ordinal,
                    len(active_cases),
                    case.case_id,
                    type(error).__name__,
                    error,
                ),
                file=sys.stderr,
            )
        finally:
            _write_manifest_atomic(manifest_path, rows)

    print(
        "Scan summary: completed=%d skipped=%d failed=%d manifest=%s"
        % (completed, skipped, failures, manifest_path)
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
