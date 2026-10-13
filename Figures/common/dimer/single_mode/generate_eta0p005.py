#!/usr/bin/env python3
"""Generate the isolated eta/nu0=0.005 data used by SI Figs. S7--S8.

The historical production scan under ``source/results/frequency_scan`` used
eta/nu0=0.01 and is intentionally left untouched.  This compact profile
reuses the same converged ED and site-local SCBA implementations, retains the
exact 369-point r mesh, and calculates only the three finite-J curves shown in
the SI.  Seven scale-invariant J=0 calculations are kept for the Fig. S7
inset.  All broadening values are retarded half-widths, not phonon damping.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wasserstein_distance


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
SOURCE = HERE / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import run_frequency_scan as scan  # noqa: E402
from single_mode_common import ModelParams  # noqa: E402


SCHEMA = "figs6_s7_eta_over_nu0_0p005_v1"
OUTPUT = HERE / "data" / "eta0p005"
SPECTRA = OUTPUT / "spectra"
MESH_PATH = OUTPUT / "mesh.csv"
BASELINE_PATH = OUTPUT / "baselines.csv"
EXTREMA_PATH = OUTPUT / "extrema.csv"
CUT_PATH = OUTPUT / "cuts.csv"
MANIFEST_PATH = OUTPUT / "manifest.csv"
VALIDATION_JSON = OUTPUT / "validation.json"
VALIDATION_TEXT = OUTPUT / "validation.txt"
TARGETED_CONVERGENCE_CSV = OUTPUT / "targeted_convergence.csv"
TARGETED_CONVERGENCE_JSON = OUTPUT / "targeted_convergence.json"
TARGETED_CONVERGENCE_TEXT = OUTPUT / "targeted_convergence.txt"

ETA_OVER_NU0 = 0.005
# The active publication arrays were calculated on this grid.  Pass it
# explicitly into the shared scan driver so import-time default binding cannot
# make the numerical grid disagree with the validation metadata.
DX = 0.002
J_ABS_MEV = 25.0
FINITE_S = (0.02, 0.05, 0.10)
BASELINE_S = (0.02, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0)
ANALYSIS_X = (-18.0, 18.0)
INTERNAL_X = (-32.0, 24.0)

MESH_COLUMNS = (
    "case_id", "huang_rhys", "j_sign", "j_meV", "r_abs", "nu0_meV",
    "g_meV", "lambda_meV", "eta_meV", "eta_over_nu0",
    "w_W1_over_nu0", "local_baseline_w", "finite_J_delta_w",
    "ed_raw_area", "scba_raw_area", "scba_raw_area_relative_difference",
    "scba_fixed_point_abs_residual_over_nu0",
    "scba_fixed_point_relative_residual", "scba_iterations",
    "scba_max_positive_im_sigma_over_nu0", "scba_channel_closure_over_nu0",
    "minimum_ed_spectrum", "minimum_scba_spectrum", "n_vib",
)
BASELINE_COLUMNS = (
    "case_id", "huang_rhys", "nu0_meV", "eta_meV", "eta_over_nu0",
    "w0_W1_over_nu0", "ed_raw_area", "scba_raw_area",
    "scba_raw_area_relative_difference",
    "scba_fixed_point_abs_residual_over_nu0",
    "scba_fixed_point_relative_residual", "scba_iterations",
    "scba_max_positive_im_sigma_over_nu0", "scba_channel_closure_over_nu0",
    "minimum_ed_spectrum", "minimum_scba_spectrum", "n_vib",
)
EXTREMA_COLUMNS = (
    "huang_rhys", "j_sign", "integer_guide", "feature_label",
    "extremum_type", "r_peak", "w_peak", "peak_case_id",
    "r_reference_min", "w_reference_min", "reference_case_id",
    "reference_label", "search_window", "reference_window",
    "verified_against_measured_mesh",
)
CUT_COLUMNS = (
    "display_order", "figure_panel", "cut_label", "role", "selection_rule",
    "case_id", "huang_rhys", "j_sign", "j_meV", "r_abs", "nu0_meV",
    "g_meV", "lambda_meV", "eta_meV", "eta_over_nu0", "w_stored",
    "w_recomputed_from_archive", "w_absolute_difference", "local_baseline_w",
    "ed_raw_area_stored", "ed_raw_area_archive", "scba_raw_area_stored",
    "scba_raw_area_archive", "archive_area_relative_difference",
    "minimum_ed_spectrum", "minimum_scba_spectrum", "archive_file",
    "archive_sha256",
)
MANIFEST_COLUMNS = (
    "kind", "case_id", "status", "path", "sha256", "size_bytes", "role",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tag(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=str(path.parent),
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
        raise


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=str(path.parent),
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
        raise


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _model(s_value: float, r_abs: float, sign: int) -> ModelParams:
    nu0 = 2.0 * J_ABS_MEV / r_abs
    return ModelParams(
        energy_00_eV=2.0,
        omega0_meV=nu0,
        huang_rhys=s_value,
        j_meV=sign * J_ABS_MEV,
        temperature_K=0.0,
        eta_meV=ETA_OVER_NU0 * nu0,
    )


def _case(s_value: float, r_abs: float, sign: int, save_archive: bool = False) -> scan.FrequencyCase:
    identifier = "eta005_S%s_J%s_r%s" % (
        _tag(s_value), "p" if sign > 0 else "m", _tag(r_abs)
    )
    return scan.FrequencyCase(
        identifier, "figs6_s7_fixed_S", _model(s_value, r_abs, sign),
        r_abs, sign, save_archive,
    )


def _baseline_case(s_value: float) -> scan.FrequencyCase:
    model = ModelParams(
        energy_00_eV=2.0, omega0_meV=50.0, huang_rhys=s_value,
        j_meV=0.0, temperature_K=0.0, eta_meV=0.25,
    )
    return scan.FrequencyCase(
        "eta005_baseline_S%s_w50" % _tag(s_value),
        "figs6_s7_J0_baseline", model, 0.0, 0, False,
    )


def _configure_scan_globals() -> None:
    # Keep run_frequency_scan's project-relative archive metadata valid even
    # though this profile lives beside, rather than below, its source module.
    scan.MODULE_DIR = HERE
    scan.DX = DX
    scan.ANALYSIS_X = ANALYSIS_X
    scan.INTERNAL_X = INTERNAL_X
    scan.SPECTRA_DIR = SPECTRA


def _core_row(case: scan.FrequencyCase, n_vib: int) -> Dict[str, Any]:
    _configure_scan_globals()
    track = scan._baseline_track(case, n_vib)
    row, _unused_sidebands = scan.run_case(
        case, n_vib, track, grid_dx=DX
    )
    return row


def _worker(payload: Tuple[float, float, int, int]) -> Dict[str, Any]:
    s_value, r_abs, sign, n_vib = payload
    case = _case(s_value, r_abs, sign, False)
    return _core_row(case, n_vib)


def _diagnostics(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "ed_raw_area": row["ed_raw_area"],
        "scba_raw_area": row["scba_raw_area"],
        "scba_raw_area_relative_difference": row["scba_raw_area_relative_difference"],
        "scba_fixed_point_abs_residual_over_nu0": row[
            "scba_fixed_point_abs_residual_over_omega0"
        ],
        "scba_fixed_point_relative_residual": row[
            "scba_fixed_point_relative_residual"
        ],
        "scba_iterations": row["scba_iterations"],
        "scba_max_positive_im_sigma_over_nu0": row[
            "scba_max_positive_im_sigma_over_omega0"
        ],
        "scba_channel_closure_over_nu0": row[
            "scba_channel_closure_over_omega0"
        ],
        "minimum_ed_spectrum": row["ed_minimum_spectrum"],
        "minimum_scba_spectrum": row["scba_minimum_spectrum"],
        "n_vib": row["n_vib"],
    }


def _baseline_output(row: Mapping[str, Any]) -> Dict[str, Any]:
    result = {
        "case_id": row["case_id"],
        "huang_rhys": row["huang_rhys"],
        "nu0_meV": row["omega0_meV"],
        "eta_meV": row["eta_meV"],
        "eta_over_nu0": row["eta_over_omega0"],
        "w0_W1_over_nu0": row["scba_w_over_omega0"],
    }
    result.update(_diagnostics(row))
    return result


def _mesh_output(row: Mapping[str, Any], baseline_w: float) -> Dict[str, Any]:
    nu0 = float(row["omega0_meV"])
    s_value = float(row["huang_rhys"])
    w_value = float(row["scba_w_over_omega0"])
    result = {
        "case_id": row["case_id"],
        "huang_rhys": s_value,
        "j_sign": int(float(row["j_sign"])),
        "j_meV": row["j_meV"],
        "r_abs": row["r_abs"],
        "nu0_meV": nu0,
        "g_meV": nu0 * math.sqrt(s_value),
        "lambda_meV": nu0 * s_value,
        "eta_meV": row["eta_meV"],
        "eta_over_nu0": row["eta_over_omega0"],
        "w_W1_over_nu0": w_value,
        "local_baseline_w": baseline_w,
        "finite_J_delta_w": w_value - baseline_w,
    }
    result.update(_diagnostics(row))
    return result


def _extrema(mesh: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for s_value in FINITE_S:
        for sign in (-1, +1):
            group = sorted(
                (row for row in mesh
                 if abs(float(row["huang_rhys"]) - s_value) < 1.0e-12
                 and int(float(row["j_sign"])) == sign),
                key=lambda row: float(row["r_abs"]),
            )
            positions = np.asarray([float(row["r_abs"]) for row in group])
            values = np.asarray([float(row["w_W1_over_nu0"]) for row in group])
            for guide in range(1, 6):
                core = np.flatnonzero(np.abs(positions - guide) <= 0.15 + 1.0e-12)
                reference = np.flatnonzero(
                    (np.abs(positions - guide) > 0.15 + 1.0e-12)
                    & (np.abs(positions - guide) <= 0.50 + 1.0e-12)
                )
                local = np.asarray([
                    index for index in range(1, len(group) - 1)
                    if abs(positions[index] - guide) < 0.50 - 1.0e-12
                    and values[index] > values[index - 1]
                    and values[index] > values[index + 1]
                ], dtype=int)
                if local.size:
                    peak = int(local[np.argmax(values[local])])
                    kind = "resolved local maximum"
                else:
                    peak = int(core[np.argmax(values[core])])
                    kind = "guide-window maximum; no resolved local cusp"
                ref = int(reference[np.argmin(values[reference])])
                outputs.append({
                    "huang_rhys": s_value,
                    "j_sign": sign,
                    "integer_guide": guide,
                    "feature_label": "odd allowed" if guide % 2 else "even forbidden control",
                    "extremum_type": kind,
                    "r_peak": positions[peak],
                    "w_peak": values[peak],
                    "peak_case_id": group[peak]["case_id"],
                    "r_reference_min": positions[ref],
                    "w_reference_min": values[ref],
                    "reference_case_id": group[ref]["case_id"],
                    "reference_label": "off-guide reference minimum",
                    "search_window": "|r-n|<0.50; core |r-n|<=0.15",
                    "reference_window": "0.15<|r-n|<=0.50",
                    "verified_against_measured_mesh": True,
                })
    return outputs


def _continuous_wasserstein(
    energy: np.ndarray, reference: np.ndarray, candidate: np.ndarray
) -> Tuple[float, float, float]:
    area_reference = float(np.trapz(reference, energy))
    area_candidate = float(np.trapz(candidate, energy))
    weights = np.empty_like(energy)
    weights[0] = 0.5 * (energy[1] - energy[0])
    weights[-1] = 0.5 * (energy[-1] - energy[-2])
    weights[1:-1] = 0.5 * (energy[2:] - energy[:-2])
    distance = float(wasserstein_distance(
        energy, energy,
        u_weights=reference / area_reference * weights,
        v_weights=candidate / area_candidate * weights,
    ))
    return distance, area_reference, area_candidate


def _select_cut_rows(
    mesh: Sequence[Mapping[str, Any]], baselines: Mapping[float, float],
    cutoffs: Mapping[float, int],
) -> List[Dict[str, Any]]:
    lookup = {str(row["case_id"]): row for row in mesh}
    outputs: List[Dict[str, Any]] = []
    order = 0
    for s_index, s_value in enumerate((0.05, 0.10)):
        group = sorted(
            (row for row in mesh
             if abs(float(row["huang_rhys"]) - s_value) < 1.0e-12
             and int(float(row["j_sign"])) == +1),
            key=lambda row: float(row["r_abs"]),
        )
        positions = np.asarray([float(row["r_abs"]) for row in group])
        values = np.asarray([float(row["w_W1_over_nu0"]) for row in group])
        local = np.asarray([
            index for index in range(1, len(group) - 1)
            if 0.50 < positions[index] < 1.50
            and values[index] > values[index - 1]
            and values[index] > values[index + 1]
        ], dtype=int)
        core = np.flatnonzero(np.abs(positions - 1.0) <= 0.15 + 1.0e-12)
        peak_index = int(local[np.argmax(values[local])]) if local.size else int(
            core[np.argmax(values[core])]
        )
        upper = np.flatnonzero(
            (positions > 1.15 + 1.0e-12) & (positions <= 1.50 + 1.0e-12)
        )
        reference_index = int(upper[np.argmin(values[upper])])
        requested = (0.65, positions[peak_index], 1.0, positions[reference_index])
        roles = (
            "lower flank", "measured maximum", "bare guide",
            "upper off-integer reference minimum",
        )
        rules = (
            "shared measured lower flank r=0.65",
            "resolved first local maximum from the measured eta/nu0=0.005 mesh",
            "measured bare first-odd commensurability guide r=1.00",
            "measured minimum on 1.15<r<=1.50",
        )
        for cut_index, (r_value, role, rule) in enumerate(zip(requested, roles, rules)):
            order += 1
            mesh_row = group[int(np.argmin(np.abs(positions - r_value)))]
            case = _case(s_value, float(mesh_row["r_abs"]), +1, True)
            core_row = _core_row(case, cutoffs[s_value])
            archive = SPECTRA / (case.case_id + ".npz")
            with np.load(archive, allow_pickle=False) as stored:
                energy = np.asarray(stored["energy_eV"], dtype=float)
                mask = np.asarray(stored["analysis_mask"], dtype=bool)
                ed = -np.imag(np.asarray(stored["ed_green_bright_per_eV"], dtype=complex))
                scba = -np.imag(np.asarray(stored["scba_green_bright_per_eV"], dtype=complex))
            distance, ed_area, scba_area = _continuous_wasserstein(
                energy[mask], ed[mask], scba[mask]
            )
            nu0 = float(mesh_row["nu0_meV"])
            recomputed = distance / (nu0 * 1.0e-3)
            stored_w = float(mesh_row["w_W1_over_nu0"])
            outputs.append({
                "display_order": order,
                "figure_panel": chr(ord("c") + s_index),
                "cut_label": "ABCD"[cut_index],
                "role": role,
                "selection_rule": rule,
                "case_id": case.case_id,
                "huang_rhys": s_value,
                "j_sign": +1,
                "j_meV": J_ABS_MEV,
                "r_abs": mesh_row["r_abs"],
                "nu0_meV": nu0,
                "g_meV": mesh_row["g_meV"],
                "lambda_meV": mesh_row["lambda_meV"],
                "eta_meV": mesh_row["eta_meV"],
                "eta_over_nu0": mesh_row["eta_over_nu0"],
                "w_stored": stored_w,
                "w_recomputed_from_archive": recomputed,
                "w_absolute_difference": abs(recomputed - stored_w),
                "local_baseline_w": baselines[s_value],
                "ed_raw_area_stored": mesh_row["ed_raw_area"],
                "ed_raw_area_archive": ed_area,
                "scba_raw_area_stored": mesh_row["scba_raw_area"],
                "scba_raw_area_archive": scba_area,
                "archive_area_relative_difference": scba_area / ed_area - 1.0,
                "minimum_ed_spectrum": float(np.min(ed[mask])),
                "minimum_scba_spectrum": float(np.min(scba[mask])),
                "archive_file": _relative(archive),
                "archive_sha256": _sha256(archive),
            })
            # The rerun must reproduce the stored mesh metric.
            if abs(float(core_row["scba_w_over_omega0"]) - stored_w) > 5.0e-10:
                raise RuntimeError("Cut rerun changed W1 for %s" % case.case_id)
    return outputs


def _validate(
    mesh: Sequence[Mapping[str, Any]], baselines: Sequence[Mapping[str, Any]],
    extrema: Sequence[Mapping[str, Any]], cuts: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def check(name: str, value: float, threshold: float, relation: str = "<=") -> None:
        passed = value <= threshold if relation == "<=" else value >= threshold
        checks.append({
            "name": name, "value": float(value), "relation": relation,
            "threshold": float(threshold), "pass": bool(passed),
        })

    check("mesh_rows", abs(len(mesh) - 2214), 0.0)
    check("unique_mesh_case_ids", abs(len({row["case_id"] for row in mesh}) - 2214), 0.0)
    check("baseline_rows", abs(len(baselines) - 7), 0.0)
    check("extrema_rows", abs(len(extrema) - 30), 0.0)
    check("cut_rows", abs(len(cuts) - 8), 0.0)
    ratios = [abs(float(row["eta_over_nu0"]) - ETA_OVER_NU0) for row in (*mesh, *baselines)]
    check("fixed_eta_over_nu0", max(ratios), 2.0e-15)
    archive_grid_residuals: List[float] = []
    archive_metadata_residuals: List[float] = []
    archive_analysis_count_errors: List[float] = []
    for row in cuts:
        archive_path = PROJECT_ROOT / str(row["archive_file"])
        with np.load(archive_path, allow_pickle=False) as archive:
            energy = np.asarray(archive["energy_eV"], dtype=float)
            mask = np.asarray(archive["analysis_mask"], dtype=bool)
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        nu0_eV = 1.0e-3 * float(row["nu0_meV"])
        archive_grid_residuals.append(
            abs(float(np.median(np.diff(energy))) / nu0_eV - DX)
        )
        archive_metadata_residuals.append(abs(
            float(metadata["dimensionless_grid"]["dx"]) - DX
        ))
        archive_analysis_count_errors.append(abs(
            int(np.count_nonzero(mask)) - 18001
        ))
    check("archive_dimensionless_grid_step", max(archive_grid_residuals), 2.0e-12)
    check("archive_grid_metadata", max(archive_metadata_residuals), 2.0e-12)
    check("archive_analysis_samples", max(archive_analysis_count_errors), 0.0)
    for label, rows in (("mesh", mesh), ("baseline", baselines)):
        check(label + "_raw_area", max(abs(float(row["scba_raw_area_relative_difference"])) for row in rows), 3.0e-6)
        check(label + "_fixed_point", max(abs(float(row["scba_fixed_point_abs_residual_over_nu0"])) for row in rows), 2.0e-12)
        check(label + "_causality", max(float(row["scba_max_positive_im_sigma_over_nu0"]) for row in rows), 2.0e-12)
        check(label + "_channel_closure", max(abs(float(row["scba_channel_closure_over_nu0"])) for row in rows), 2.0e-12)
        check(label + "_ED_positivity", min(float(row["minimum_ed_spectrum"]) for row in rows), -1.0e-12, ">=")
        check(label + "_SCBA_positivity", min(float(row["minimum_scba_spectrum"]) for row in rows), -1.0e-12, ">=")
    # Archives store complex Green functions as complex64, as in the original
    # production workflow.  On the 28,001-point internal grid the accumulated
    # quadrature roundoff is at most 2.1e-9 in dimensionless W1.
    check("archive_W1_reproduction", max(float(row["w_absolute_difference"]) for row in cuts), 3.0e-9)
    check("archive_area_reproduction", max(
        abs(float(row["ed_raw_area_archive"]) / float(row["ed_raw_area_stored"]) - 1.0)
        for row in cuts
    ), 5.0e-8)
    check("archive_SCBA_area_reproduction", max(
        abs(float(row["scba_raw_area_archive"]) / float(row["scba_raw_area_stored"]) - 1.0)
        for row in cuts
    ), 5.0e-8)
    check("archive_ED_positivity", min(float(row["minimum_ed_spectrum"]) for row in cuts), -1.0e-10, ">=")
    check("archive_SCBA_positivity", min(float(row["minimum_scba_spectrum"]) for row in cuts), -1.0e-10, ">=")
    for s_value in FINITE_S:
        for sign in (-1, +1):
            rows = [row for row in mesh if abs(float(row["huang_rhys"]) - s_value) < 1.0e-12 and int(float(row["j_sign"])) == sign]
            check("mesh_count_S%s_J%s" % (_tag(s_value), "p" if sign > 0 else "m"), abs(len(rows) - 369), 0.0)
    payload = {
        "schema_version": SCHEMA,
        "generated_utc": _utc_now(),
        "physical_profile": {
            "J_abs_meV": J_ABS_MEV,
            "temperature_K": 0.0,
            "eta_over_nu0": ETA_OVER_NU0,
            "dimensionless_step": DX,
            "internal_grid_samples": 28001,
            "analysis_grid_samples": 18001,
            "analysis_x": ANALYSIS_X,
            "internal_x": INTERNAL_X,
            "finite_J_S": FINITE_S,
            "baseline_S": BASELINE_S,
        },
        "ed_profile": {
            "basis": (
                "factorized common/difference oscillator Fock bases; "
                "common mode analytic and difference mode diagonalized"
            ),
            "finite_J_n_vib": 10,
            "finite_J_difference_matrix_dimension": 20,
            "finite_J_parity_block_dimension": 10,
            "baseline_n_vib_by_S": {
                "0.02": 10, "0.05": 10, "0.10": 10, "0.25": 12,
                "0.50": 16, "1.0": 20, "2.0": 24,
            },
        },
        "scba_profile": {
            "production_solver": "zero_temperature_ascending_sweep",
            "production_iterations": 1,
            "absolute_residual_tolerance_eV": 1.0e-9,
            "relative_residual_tolerance": 1.0e-8,
            "green_change_tolerance": 1.0e-8,
            "maximum_iterations_for_iterative_fallback": 5000,
            "mixing": {
                "used_by_production_T0_sweep": False,
                "initial": 0.08,
                "minimum": 0.002,
                "maximum": 0.40,
                "growth_factor": 1.05,
                "halve_if_residual_growth_factor_exceeds": 1.10,
                "grow_if_residual_ratio_below": 0.70,
                "consecutive_converged": 3,
            },
            "causality_tolerance_eV": 5.0e-10,
            "profile_validation_tolerance_over_nu0": 2.0e-12,
            "raw_area_relative_tolerance": 3.0e-6,
        },
        "wasserstein_protocol": {
            "common_support_x": ANALYSIS_X,
            "normalization": "independent trapezoidal unit-area normalization",
            "integration": (
                "weighted one-dimensional Wasserstein distance with "
                "trapezoidal point weights, equivalent to CDF integration"
            ),
            "smoothing": False,
            "peak_normalization": False,
        },
        "completion_protocol": {
            "expected_finite_J_cases": 2214,
            "expected_J0_baselines": 7,
            "failed_or_nonconverged_points_dropped": False,
            "failure_behavior": "worker exception aborts generation",
        },
        "targeted_convergence": {
            "command": (
                "python3 Figures/common/dimer/single_mode/"
                "validate_targeted_convergence.py"
            ),
            "csv": _relative(TARGETED_CONVERGENCE_CSV),
            "json": _relative(TARGETED_CONVERGENCE_JSON),
            "text": _relative(TARGETED_CONVERGENCE_TEXT),
        },
        "counts": {
            "mesh": len(mesh), "baselines": len(baselines),
            "extrema": len(extrema), "cuts": len(cuts),
        },
        "checks": checks,
        "pass": bool(all(item["pass"] for item in checks)),
    }
    return payload


def _manifest(cuts: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    roles = {
        MESH_PATH: "2,214 freshly calculated finite-J mesh points",
        BASELINE_PATH: "seven freshly calculated scale-invariant J=0 baselines",
        EXTREMA_PATH: "extrema measured directly from the eta/nu0=0.005 mesh",
        CUT_PATH: "eight A-D cut selections and archive checks",
        VALIDATION_JSON: "machine-readable numerical validation",
        VALIDATION_TEXT: "human-readable PASS report",
        Path(__file__).resolve(): "reproducible eta/nu0=0.005 generator",
    }
    optional_validation = {
        TARGETED_CONVERGENCE_CSV:
            "three-case ED-cutoff and grid-halving convergence data",
        TARGETED_CONVERGENCE_JSON:
            "machine-readable targeted convergence summary",
        TARGETED_CONVERGENCE_TEXT:
            "human-readable targeted convergence PASS report",
    }
    roles.update({
        path: role for path, role in optional_validation.items() if path.is_file()
    })
    rows: List[Dict[str, Any]] = []
    for path, role in roles.items():
        rows.append({
            "kind": "source" if path == Path(__file__).resolve() else "data",
            "case_id": "", "status": "complete", "path": _relative(path),
            "sha256": _sha256(path), "size_bytes": path.stat().st_size,
            "role": role,
        })
    for cut in cuts:
        path = PROJECT_ROOT / str(cut["archive_file"])
        rows.append({
            "kind": "archive", "case_id": cut["case_id"], "status": "complete",
            "path": cut["archive_file"], "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
            "role": "S=%g cut %s" % (float(cut["huang_rhys"]), cut["cut_label"]),
        })
    return sorted(rows, key=lambda row: (str(row["kind"]), str(row["path"])))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--checkpoint", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers < 1 or args.checkpoint < 1:
        raise ValueError("--workers and --checkpoint must be positive")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SPECTRA.mkdir(parents=True, exist_ok=True)
    _configure_scan_globals()

    # The cutoff criterion depends only on the ED sticks, not eta or DX.
    cutoff_map, _cutoff_rows = scan.determine_primary_cutoffs()
    baselines_raw: List[Dict[str, Any]] = []
    for s_value in BASELINE_S:
        case = _baseline_case(s_value)
        baselines_raw.append(_core_row(case, cutoff_map[s_value]))
    baselines = [_baseline_output(row) for row in baselines_raw]
    baselines.sort(key=lambda row: float(row["huang_rhys"]))
    baseline_by_s = {
        float(row["huang_rhys"]): float(row["w0_W1_over_nu0"])
        for row in baselines
    }
    _write_csv(BASELINE_PATH, baselines, BASELINE_COLUMNS)

    existing = {} if args.overwrite else {
        row["case_id"]: row for row in _read_csv(MESH_PATH)
    }
    payloads: List[Tuple[float, float, int, int]] = []
    for s_value in FINITE_S:
        for sign in (-1, +1):
            for r_abs in scan.primary_r_grid():
                identifier = _case(s_value, r_abs, sign).case_id
                if identifier not in existing:
                    payloads.append((s_value, r_abs, sign, cutoff_map[s_value]))
    print("eta/nu0=0.005 scan: pending=%d existing=%d workers=%d" % (
        len(payloads), len(existing), args.workers
    ), flush=True)

    completed_since_checkpoint = 0
    if payloads:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(_worker, payload): payload for payload in payloads}
            for ordinal, future in enumerate(as_completed(future_map), start=1):
                raw = future.result()
                s_value = float(raw["huang_rhys"])
                output = _mesh_output(raw, baseline_by_s[s_value])
                existing[str(output["case_id"])] = output
                completed_since_checkpoint += 1
                if completed_since_checkpoint >= args.checkpoint:
                    ordered = sorted(existing.values(), key=lambda row: (
                        float(row["huang_rhys"]), int(float(row["j_sign"])),
                        float(row["r_abs"]),
                    ))
                    _write_csv(MESH_PATH, ordered, MESH_COLUMNS)
                    completed_since_checkpoint = 0
                if ordinal % 100 == 0 or ordinal == len(payloads):
                    print("completed %d/%d" % (ordinal, len(payloads)), flush=True)
    mesh = sorted(existing.values(), key=lambda row: (
        float(row["huang_rhys"]), int(float(row["j_sign"])), float(row["r_abs"]),
    ))
    _write_csv(MESH_PATH, mesh, MESH_COLUMNS)

    extrema = _extrema(mesh)
    _write_csv(EXTREMA_PATH, extrema, EXTREMA_COLUMNS)
    cuts = _select_cut_rows(mesh, baseline_by_s, cutoff_map)
    _write_csv(CUT_PATH, cuts, CUT_COLUMNS)
    validation = _validate(mesh, baselines, extrema, cuts)
    _write_json(VALIDATION_JSON, validation)
    lines = [
        "Fig. S7-S8 eta/nu0=0.005 numerical validation",
        "generated_utc=%s" % validation["generated_utc"],
        "production_grid=x in [-32,24], Delta x=0.002, points=28001",
        "analysis_support=x in [-18,18], points=18001",
        "SCBA_solver=zero_temperature_ascending_sweep, iterations=1",
        "SCBA_tolerances=absolute 1e-9 eV, relative 1e-8; "
        "iterative_fallback_max_iterations=5000",
        "W1=independent trapezoidal normalization and weighted CDF-equivalent "
        "integration; smoothing=none; peak_normalization=none",
        "completion=2214 finite-J plus 7 J=0; worker failure aborts",
    ]
    for item in validation["checks"]:
        lines.append("%s %s value=%.12e threshold %s %.12e" % (
            "PASS" if item["pass"] else "FAIL", item["name"], item["value"],
            item["relation"], item["threshold"],
        ))
    lines.append("status = %s" % ("PASS" if validation["pass"] else "FAIL"))
    lines.append("failures = %d" % sum(not item["pass"] for item in validation["checks"]))
    VALIDATION_TEXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_csv(MANIFEST_PATH, _manifest(cuts), MANIFEST_COLUMNS)
    print("validation=%s mesh=%d baselines=%d cuts=%d" % (
        "PASS" if validation["pass"] else "FAIL", len(mesh), len(baselines), len(cuts)
    ), flush=True)
    return 0 if validation["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
