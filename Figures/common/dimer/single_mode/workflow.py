#!/usr/bin/env python3
"""Shared validation and plotting for SI Figs. S7--S8.

The two numbered figure drivers import this module.  It reads the active
``eta/nu0=0.005`` figure profile through ``data/source_manifest.json``,
independently verifies its 2,214-point mesh, seven baselines, and eight selected
archives, and exposes separate plotting functions for the full error scan and
the A--D spectral cuts.  A legacy loader remains for the preserved historical
``eta/nu0=0.01`` production archive.

Lines join calculated points in their measured order.  No interpolation,
spline, smoothing, fitting, or peak normalization is performed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_ROOT = SCRIPT_DIR.parents[3]
SOURCE_MANIFEST_PATH = SCRIPT_DIR / "data" / "source_manifest.json"
COMMON_DIR = SCRIPT_DIR.parents[1]
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from si_plot_style import (  # noqa: E402
    EXACT_STYLE,
    SCBA_STYLE,
    S_COLORS,
    add_panel_label,
    compact_legend_kwargs,
    configure_si_matplotlib,
    style_axis,
)

SELECTED_S = (0.02, 0.05, 0.10)
ALL_S = (0.02, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0)
SIGNS = (+1, -1)
J_ABS_MEV = 25.0
ETA_OVER_NU = 0.005
PRODUCTION_DX = 0.002

BASELINE_DASH = (0, (5.0, 2.2))
CUT_LABELS = ("A", "B", "C", "D")
SWEEP_S = (0.05, 0.10)
SWEEP_R_WINDOW = (0.60, 1.50)
SPECTRAL_X_WINDOW = (-1.0, 1.5)

MESH_COLUMNS = (
    "case_id", "huang_rhys", "j_sign", "j_meV", "r_abs", "nu0_meV",
    "g_meV", "lambda_meV", "eta_meV", "eta_over_nu0",
    "w_W1_over_nu0", "local_baseline_w", "finite_J_delta_w",
)
BASELINE_COLUMNS = (
    "case_id", "huang_rhys", "nu0_meV", "eta_meV", "eta_over_nu0",
    "w0_W1_over_nu0", "ed_raw_area", "scba_raw_area",
    "scba_raw_area_relative_difference",
)
EXTREMA_COLUMNS = (
    "huang_rhys", "j_sign", "integer_guide", "feature_label",
    "extremum_type", "r_peak", "w_peak", "peak_case_id",
    "r_reference_min", "w_reference_min", "reference_case_id",
    "reference_label", "search_window", "reference_window",
    "verified_against_measured_mesh",
)
CUT_COLUMNS = (
    "display_order", "figure_panel", "cut_label", "role",
    "selection_rule", "case_id", "huang_rhys", "j_sign", "j_meV",
    "r_abs", "nu0_meV", "g_meV", "lambda_meV", "eta_meV", "eta_over_nu0",
    "w_stored", "w_recomputed_from_archive", "w_absolute_difference",
    "local_baseline_w", "ed_raw_area_stored", "ed_raw_area_archive",
    "scba_raw_area_stored", "scba_raw_area_archive",
    "archive_area_relative_difference", "minimum_ed_spectrum",
    "minimum_scba_spectrum", "archive_file", "archive_sha256",
)
PROVENANCE_COLUMNS = ("kind", "path", "sha256", "size_bytes", "role")


def _configure_style() -> None:
    configure_si_matplotlib()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError("Required CSV is missing: %s" % path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header: %s" % path)
        return [dict(row) for row in reader]


def _write_csv_atomic(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=str(path.parent),
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            writer = csv.DictWriter(
                handle, fieldnames=list(columns), extrasaction="ignore"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
    except Exception:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
        raise


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        os.replace(temporary, str(path))
    except Exception:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
        raise


def _float(row: Mapping[str, Any], key: str) -> float:
    text = str(row.get(key, "")).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _close(left: float, right: float, tolerance: float = 1.0e-10) -> bool:
    return bool(np.isfinite(left) and np.isfinite(right)
                and abs(left - right) <= tolerance)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _project_relative(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve()))


def _panel_label(axis: plt.Axes, label: str) -> None:
    add_panel_label(axis, "(%s)" % label)


def _quadrature_weights(x: np.ndarray) -> np.ndarray:
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("Wasserstein quadrature requires an increasing grid")
    weights = np.empty_like(x)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def _continuous_wasserstein(
    energy_eV: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
) -> Tuple[float, float, float]:
    reference_area = float(np.trapz(reference, energy_eV))
    candidate_area = float(np.trapz(candidate, energy_eV))
    if reference_area <= 0.0 or candidate_area <= 0.0:
        raise ValueError("Wasserstein inputs must have positive raw areas")
    quadrature = _quadrature_weights(energy_eV)
    distance = float(wasserstein_distance(
        energy_eV, energy_eV,
        u_weights=(reference / reference_area) * quadrature,
        v_weights=(candidate / candidate_area) * quadrature,
    ))
    return distance, reference_area, candidate_area


def _finite_values(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = np.asarray([_float(row, key) for row in rows], dtype=float)
    if values.size != len(rows) or not np.all(np.isfinite(values)):
        raise ValueError("Column %s contains missing/nonfinite values" % key)
    return values


def _validate_global_summary(summary: Sequence[Dict[str, str]]) -> Dict[str, float]:
    ed_areas = _finite_values(summary, "ed_raw_area")
    scba_areas = _finite_values(summary, "scba_raw_area")
    raw_mismatch = np.abs(_finite_values(
        summary, "scba_raw_area_relative_difference"
    ))
    residual = np.abs(_finite_values(
        summary, "scba_fixed_point_abs_residual_over_omega0"
    ))
    positive_im_sigma = _finite_values(
        summary, "scba_max_positive_im_sigma_over_omega0"
    )
    channel_closure = np.abs(_finite_values(
        summary, "scba_channel_closure_over_omega0"
    ))
    ed_minimum = _finite_values(summary, "ed_minimum_spectrum")
    scba_minimum = _finite_values(summary, "scba_minimum_spectrum")

    checks = {
        "max_raw_area_relative_mismatch": float(np.max(raw_mismatch)),
        "max_scba_fixed_point_residual_over_nu0": float(np.max(residual)),
        "max_positive_im_sigma_over_nu0": float(np.max(positive_im_sigma)),
        "max_channel_closure_over_nu0": float(np.max(channel_closure)),
        "minimum_ed_spectrum": float(np.min(ed_minimum)),
        "minimum_scba_spectrum": float(np.min(scba_minimum)),
    }
    if float(np.min(ed_areas)) <= 0.0 or float(np.min(scba_areas)) <= 0.0:
        raise ValueError("A production spectrum has a nonpositive raw area")
    if checks["max_raw_area_relative_mismatch"] > 3.0e-6:
        raise ValueError("Production raw-area tolerance failed")
    if checks["max_scba_fixed_point_residual_over_nu0"] > 2.0e-12:
        raise ValueError("Production SCBA fixed-point tolerance failed")
    if checks["max_positive_im_sigma_over_nu0"] > 2.0e-12:
        raise ValueError("Production retarded-causality tolerance failed")
    if checks["max_channel_closure_over_nu0"] > 2.0e-12:
        raise ValueError("Production channel-closure tolerance failed")
    if checks["minimum_ed_spectrum"] < -1.0e-12:
        raise ValueError("Production ED positivity tolerance failed")
    if checks["minimum_scba_spectrum"] < -1.0e-12:
        raise ValueError("Production SCBA positivity tolerance failed")
    return checks


def _source_paths(project_root: Path) -> Tuple[Path, Dict[str, Path]]:
    """Resolve and validate the project-relative shared-data manifest."""

    if not SOURCE_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            "Single-mode source manifest is missing: %s" % SOURCE_MANIFEST_PATH
        )
    with SOURCE_MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if int(manifest.get("schema_version", -1)) not in (1, 2):
        raise ValueError("Unexpected single-mode source-manifest schema")
    source_root_text = str(manifest.get("source_root", "")).strip()
    file_map = manifest.get("files")
    if not source_root_text or Path(source_root_text).is_absolute():
        raise ValueError("source_root must be a nonempty project-relative path")
    if not isinstance(file_map, dict):
        raise ValueError("source-manifest files must be a mapping")
    source_root = (project_root / source_root_text).resolve()
    try:
        source_root.relative_to(project_root.resolve())
    except ValueError:
        raise ValueError("source_root escapes the project")
    if not source_root.is_dir():
        raise FileNotFoundError("Relocated single-mode source is missing: %s" % source_root)
    required = (
        "manifest", "summary", "line_scan", "baselines", "extrema", "cuts",
        "validation_json", "final_audit", "targeted_convergence_csv",
        "targeted_convergence_json", "targeted_convergence_text",
    )
    paths: Dict[str, Path] = {"source_manifest": SOURCE_MANIFEST_PATH.resolve()}
    for key in required:
        relative_text = str(file_map.get(key, "")).strip()
        if not relative_text or Path(relative_text).is_absolute():
            raise ValueError("Manifest entry %s must be project-relative" % key)
        candidate = (source_root / relative_text).resolve()
        try:
            candidate.relative_to(source_root)
        except ValueError:
            raise ValueError("Manifest entry %s escapes source_root" % key)
        if not candidate.is_file():
            raise FileNotFoundError("Manifest entry %s is missing: %s" % (key, candidate))
        paths[key] = candidate
    return source_root, paths


def _load_eta0p005_profile(
    project_root: Path,
    sources: Mapping[str, Path],
    validation: Mapping[str, Any],
) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]],
    List[Dict[str, Any]], Dict[str, float], Dict[str, Path],
]:
    """Load and independently recheck the fresh Fig. S7--S8 profile."""

    if validation.get("schema_version") != "figs6_s7_eta_over_nu0_0p005_v1":
        raise ValueError("Unexpected eta/nu0=0.005 profile schema")
    if not bool(validation.get("pass")):
        raise RuntimeError("The fresh eta/nu0=0.005 validation is not PASS")
    audit_text = sources["final_audit"].read_text(encoding="utf-8")
    if "status = PASS" not in audit_text or "failures = 0" not in audit_text:
        raise RuntimeError("The fresh eta/nu0=0.005 text audit is not PASS")

    inventory = _read_csv(sources["manifest"])
    if len(inventory) != 18 or any(row.get("status") != "complete" for row in inventory):
        raise ValueError("Expected 18 complete profile inventory entries")
    for row in inventory:
        text = str(row.get("path", ""))
        if not text or Path(text).is_absolute():
            raise ValueError("Profile inventory paths must be project-relative")
        path = (project_root / text).resolve()
        try:
            path.relative_to(project_root)
        except ValueError:
            raise ValueError("Profile inventory path escapes the project")
        if (not path.is_file()
                or _sha256(path) != str(row.get("sha256", ""))
                or path.stat().st_size != int(row.get("size_bytes", "-1"))):
            raise ValueError("Profile inventory hash/size mismatch: %s" % path)

    mesh_rows: List[Dict[str, Any]] = _read_csv(sources["line_scan"])
    baseline_rows: List[Dict[str, Any]] = _read_csv(sources["baselines"])
    extrema_rows: List[Dict[str, Any]] = _read_csv(sources["extrema"])
    cuts_source: List[Dict[str, Any]] = _read_csv(sources["cuts"])
    if (len(mesh_rows), len(baseline_rows), len(extrema_rows), len(cuts_source)) != (
        2214, 7, 30, 8,
    ):
        raise ValueError("Fresh profile row counts changed")
    identifiers = [str(row.get("case_id", "")) for row in mesh_rows]
    if "" in identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Fresh mesh case IDs are empty or duplicated")

    baseline_by_s = {
        round(_float(row, "huang_rhys"), 12): row for row in baseline_rows
    }
    if tuple(sorted(baseline_by_s)) != ALL_S:
        raise ValueError("Fresh J=0 baseline S grid changed")
    shared_r: Optional[np.ndarray] = None
    for s_value in SELECTED_S:
        for sign in (-1, +1):
            rows = sorted(
                (row for row in mesh_rows
                 if _close(_float(row, "huang_rhys"), s_value)
                 and int(round(_float(row, "j_sign"))) == sign),
                key=lambda row: _float(row, "r_abs"),
            )
            if len(rows) != 369:
                raise ValueError("Expected 369 fresh points for S=%g, J sign=%+d" % (s_value, sign))
            r_values = np.asarray([_float(row, "r_abs") for row in rows])
            if (not _close(float(r_values[0]), 0.25)
                    or not _close(float(r_values[-1]), 5.5)
                    or np.any(np.diff(r_values) <= 0.0)):
                raise ValueError("Fresh r mesh range/order changed")
            if shared_r is None:
                shared_r = r_values
            elif not np.array_equal(shared_r, r_values):
                raise ValueError("Fresh S/sign groups do not share one r mesh")
            for row in rows:
                r_abs = _float(row, "r_abs")
                nu0 = _float(row, "nu0_meV")
                if (not _close(abs(_float(row, "j_meV")), J_ABS_MEV)
                        or not _close(nu0, 2.0 * J_ABS_MEV / r_abs, 2.0e-10)
                        or not _close(_float(row, "eta_over_nu0"), ETA_OVER_NU, 2.0e-15)
                        or not _close(_float(row, "eta_meV"), ETA_OVER_NU * nu0, 2.0e-12)):
                    raise ValueError("Fresh physical profile changed for %s" % row["case_id"])
                baseline = _float(baseline_by_s[round(s_value, 12)], "w0_W1_over_nu0")
                if (not _close(_float(row, "local_baseline_w"), baseline, 2.0e-12)
                        or not _close(
                            _float(row, "finite_J_delta_w"),
                            _float(row, "w_W1_over_nu0") - baseline,
                            2.0e-12,
                        )):
                    raise ValueError("Fresh baseline join changed for %s" % row["case_id"])
    for row in baseline_rows:
        if (not _close(_float(row, "nu0_meV"), 50.0)
                or not _close(_float(row, "eta_meV"), 0.25)
                or not _close(_float(row, "eta_over_nu0"), ETA_OVER_NU, 2.0e-15)):
            raise ValueError("Fresh J=0 baseline resolution changed")

    all_calculated = list(mesh_rows) + list(baseline_rows)
    global_checks = {
        "profile_case_count": float(len(all_calculated)),
        "max_raw_area_relative_mismatch": float(max(
            abs(_float(row, "scba_raw_area_relative_difference"))
            for row in all_calculated
        )),
        "max_scba_fixed_point_residual_over_nu0": float(max(
            abs(_float(row, "scba_fixed_point_abs_residual_over_nu0"))
            for row in all_calculated
        )),
        "max_positive_im_sigma_over_nu0": float(max(
            _float(row, "scba_max_positive_im_sigma_over_nu0")
            for row in all_calculated
        )),
        "max_channel_closure_over_nu0": float(max(
            abs(_float(row, "scba_channel_closure_over_nu0"))
            for row in all_calculated
        )),
        "minimum_ed_spectrum": float(min(
            _float(row, "minimum_ed_spectrum") for row in all_calculated
        )),
        "minimum_scba_spectrum": float(min(
            _float(row, "minimum_scba_spectrum") for row in all_calculated
        )),
    }
    if global_checks["max_raw_area_relative_mismatch"] > 3.0e-6:
        raise ValueError("Fresh raw-area tolerance failed")
    if global_checks["max_scba_fixed_point_residual_over_nu0"] > 2.0e-12:
        raise ValueError("Fresh SCBA fixed-point tolerance failed")
    if (global_checks["max_positive_im_sigma_over_nu0"] > 2.0e-12
            or global_checks["max_channel_closure_over_nu0"] > 2.0e-12):
        raise ValueError("Fresh causality/channel tolerance failed")
    if (global_checks["minimum_ed_spectrum"] < -1.0e-12
            or global_checks["minimum_scba_spectrum"] < -1.0e-12):
        raise ValueError("Fresh spectral-positivity tolerance failed")

    # Verify extrema against the measured mesh instead of trusting labels.
    mesh_groups: Dict[Tuple[float, int], List[Dict[str, Any]]] = {}
    for s_value in SELECTED_S:
        for sign in (-1, +1):
            mesh_groups[(s_value, sign)] = sorted(
                (row for row in mesh_rows
                 if _close(_float(row, "huang_rhys"), s_value)
                 and int(round(_float(row, "j_sign"))) == sign),
                key=lambda row: _float(row, "r_abs"),
            )
    for row in extrema_rows:
        group = mesh_groups[(_float(row, "huang_rhys"), int(round(_float(row, "j_sign"))))]
        peak = next(item for item in group if item["case_id"] == row["peak_case_id"])
        reference = next(item for item in group if item["case_id"] == row["reference_case_id"])
        if (not _close(_float(row, "r_peak"), _float(peak, "r_abs"))
                or not _close(_float(row, "w_peak"), _float(peak, "w_W1_over_nu0"), 2.0e-12)
                or not _close(_float(row, "r_reference_min"), _float(reference, "r_abs"))
                or not _close(_float(row, "w_reference_min"), _float(reference, "w_W1_over_nu0"), 2.0e-12)):
            raise ValueError("Fresh measured-extremum record changed")

    cut_rows: List[Dict[str, Any]] = []
    expected_order = 0
    for source in sorted(cuts_source, key=lambda row: int(round(_float(row, "display_order")))):
        expected_order += 1
        if (int(round(_float(source, "display_order"))) != expected_order
                or source.get("cut_label") != CUT_LABELS[(expected_order - 1) % 4]
                or not _close(_float(source, "eta_over_nu0"), ETA_OVER_NU, 2.0e-15)):
            raise ValueError("Fresh A--D ordering/profile changed")
        archive_text = str(source.get("archive_file", ""))
        archive_path = (project_root / archive_text).resolve()
        try:
            archive_path.relative_to(project_root)
        except ValueError:
            raise ValueError("Fresh cut archive escapes the project")
        if (not archive_path.is_file()
                or _sha256(archive_path) != str(source.get("archive_sha256", ""))):
            raise ValueError("Fresh cut archive hash changed: %s" % archive_path)
        with np.load(str(archive_path), allow_pickle=False) as archive:
            required = (
                "energy_eV", "analysis_mask", "ed_green_bright_per_eV",
                "scba_green_bright_per_eV", "metadata_json",
            )
            missing = [key for key in required if key not in archive.files]
            if missing:
                raise KeyError("Fresh cut archive lacks %s" % missing)
            energy = np.asarray(archive["energy_eV"], dtype=float)
            mask = np.asarray(archive["analysis_mask"], dtype=bool)
            ed_green = np.asarray(archive["ed_green_bright_per_eV"], dtype=complex)
            scba_green = np.asarray(archive["scba_green_bright_per_eV"], dtype=complex)
            metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
        if (metadata.get("case_id") != source.get("case_id")
                or energy.ndim != 1 or mask.shape != energy.shape
                or ed_green.shape != energy.shape or scba_green.shape != energy.shape
                or np.any(np.diff(energy) <= 0.0)
                or not np.all(np.isfinite(ed_green))
                or not np.all(np.isfinite(scba_green))):
            raise ValueError("Fresh cut archive is malformed")
        model_metadata = metadata.get("model", {})
        grid_metadata = metadata.get("grid", {})
        dimensionless_metadata = metadata.get("dimensionless_grid", {})
        nu0_eV = 1.0e-3 * float(
            model_metadata.get("omega0_meV", float("nan"))
        )
        measured_dx = float(np.median(np.diff(energy))) / nu0_eV
        if (not _close(measured_dx, PRODUCTION_DX, 2.0e-12)
                or not _close(
                    float(dimensionless_metadata.get("dx", float("nan"))),
                    PRODUCTION_DX,
                    2.0e-12,
                )
                or not _close(
                    1.0e-3 * float(grid_metadata.get("step_meV", float("nan")))
                    / nu0_eV,
                    PRODUCTION_DX,
                    2.0e-12,
                )
                or int(np.count_nonzero(mask)) != 18001):
            raise ValueError("Fresh cut archive grid metadata changed")
        selected_energy = energy[mask]
        ed_spectrum = -np.imag(ed_green[mask])
        scba_spectrum = -np.imag(scba_green[mask])
        distance, ed_area, scba_area = _continuous_wasserstein(
            selected_energy, ed_spectrum, scba_spectrum
        )
        recomputed = distance / (_float(source, "nu0_meV") * 1.0e-3)
        if (abs(recomputed - _float(source, "w_stored")) > 3.0e-9
                or abs(recomputed - _float(source, "w_recomputed_from_archive")) > 2.0e-12
                or abs(ed_area / _float(source, "ed_raw_area_stored") - 1.0) > 5.0e-8
                or abs(scba_area / _float(source, "scba_raw_area_stored") - 1.0) > 5.0e-8
                or float(np.min(ed_spectrum)) < -1.0e-10
                or float(np.min(scba_spectrum)) < -1.0e-10):
            raise ValueError("Fresh cut archive numerical check failed")
        output: Dict[str, Any] = dict(source)
        output.update({
            "_energy_eV": energy,
            "_analysis_mask": mask,
            "_ed_spectrum": -np.imag(ed_green),
            "_scba_spectrum": -np.imag(scba_green),
            "_metadata": metadata,
        })
        cut_rows.append(output)
    global_checks["max_archive_w_absolute_difference"] = float(max(
        _float(row, "w_absolute_difference") for row in cut_rows
    ))
    global_checks["production_dimensionless_grid_step"] = PRODUCTION_DX
    return (
        mesh_rows, baseline_rows, extrema_rows, cut_rows,
        global_checks, dict(sources),
    )


def load_validated_data(
    project_root: Path,
) -> Tuple[
    List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]],
    List[Dict[str, Any]], Dict[str, float], Dict[str, Path],
]:
    project_root = project_root.resolve()
    bath_dir, sources = _source_paths(project_root)
    with sources["validation_json"].open("r", encoding="utf-8") as handle:
        routed_validation = json.load(handle)
    if routed_validation.get("schema_version") == "figs6_s7_eta_over_nu0_0p005_v1":
        return _load_eta0p005_profile(
            project_root, sources, routed_validation
        )
    manifest = _read_csv(sources["manifest"])
    summary = _read_csv(sources["summary"])
    if len(manifest) != 5251 or len(summary) != 5251:
        raise ValueError("Expected 5251 manifest and summary rows")
    if any(row.get("status") != "complete" for row in manifest):
        raise RuntimeError("The frequency-scan manifest is not wholly complete")
    manifest_ids = [row.get("case_id", "") for row in manifest]
    summary_ids = [row.get("case_id", "") for row in summary]
    if ("" in manifest_ids or "" in summary_ids
            or len(set(manifest_ids)) != len(manifest_ids)
            or len(set(summary_ids)) != len(summary_ids)
            or set(manifest_ids) != set(summary_ids)):
        raise ValueError("Manifest/summary case IDs are empty, duplicate, or unequal")
    summary_by_id = {row["case_id"]: row for row in summary}

    with sources["validation_json"].open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    if not bool(validation.get("pass")):
        raise RuntimeError("The authoritative frequency-scan validation is not PASS")
    audit_text = sources["final_audit"].read_text(encoding="utf-8")
    if "status = PASS" not in audit_text or "failures = 0" not in audit_text:
        raise RuntimeError("The independent final audit is not PASS")
    global_checks = _validate_global_summary(summary)

    source_mesh = _read_csv(sources["line_scan"])
    mesh_rows: List[Dict[str, Any]] = []
    source_mesh_by_id = {row.get("case_id", ""): row for row in source_mesh}
    if len(source_mesh) != 2214 or len(source_mesh_by_id) != len(source_mesh):
        raise ValueError("Expected 2214 unique active line-scan rows")

    baselines_source = [
        row for row in summary if row.get("family") == "primary_J0_baseline"
    ]
    if len(baselines_source) != len(ALL_S):
        raise ValueError("Expected seven fixed-S local baselines")
    baseline_by_s = {
        round(_float(row, "huang_rhys"), 12): row for row in baselines_source
    }
    if tuple(sorted(baseline_by_s)) != ALL_S:
        raise ValueError("The local-baseline S grid changed")

    shared_r: Optional[np.ndarray] = None
    for s_value in SELECTED_S:
        for sign in (-1, +1):
            rows = sorted(
                (
                    row for row in summary
                    if row.get("family") == "primary_fixed_S"
                    and _close(_float(row, "huang_rhys"), s_value)
                    and int(round(_float(row, "j_sign"))) == sign
                ),
                key=lambda row: _float(row, "r_abs"),
            )
            if len(rows) != 369:
                raise ValueError(
                    "Expected 369 measured r points for S=%g sign=%+d" %
                    (s_value, sign)
                )
            r_values = np.asarray([_float(row, "r_abs") for row in rows])
            if not (_close(float(r_values[0]), 0.25)
                    and _close(float(r_values[-1]), 5.5)
                    and np.all(np.diff(r_values) > 0.0)):
                raise ValueError("Measured r mesh has invalid range/order")
            for guide in range(1, 6):
                expected_refinement = np.arange(
                    guide - 0.15, guide + 0.1500001, 0.005
                )
                for requested in expected_refinement:
                    if not np.any(np.isclose(
                        r_values, requested, rtol=0.0, atol=1.0e-10
                    )):
                        raise ValueError(
                            "Missing measured refined point r=%.3f" % requested
                        )
            if shared_r is None:
                shared_r = r_values
            elif not np.array_equal(shared_r, r_values):
                raise ValueError("Selected S/sign groups do not share one r mesh")

            baseline = baseline_by_s[round(s_value, 12)]
            baseline_w = _float(baseline, "scba_w_over_omega0")
            for row in rows:
                case_id = row["case_id"]
                source = source_mesh_by_id.get(case_id)
                if source is None:
                    raise KeyError("Line-scan source omits %s" % case_id)
                for key in (
                    "huang_rhys", "j_sign", "j_meV", "r_abs",
                    "omega0_meV", "eta_over_omega0", "scba_w_over_omega0",
                ):
                    if not _close(_float(row, key), _float(source, key), 2.0e-12):
                        raise ValueError(
                            "Line-source/summary mismatch for %s:%s" %
                            (case_id, key)
                        )
                r_abs = _float(row, "r_abs")
                nu0 = _float(row, "omega0_meV")
                if not _close(nu0, 2.0 * J_ABS_MEV / r_abs, 2.0e-10):
                    raise ValueError("Frequency/r relation failed for %s" % case_id)
                if not _close(abs(_float(row, "j_meV")), J_ABS_MEV):
                    raise ValueError("The fixed-|J| scan changed")
                if not _close(_float(row, "eta_over_omega0"), ETA_OVER_NU):
                    raise ValueError("The fixed eta/nu0 scan changed")
                w_value = _float(row, "scba_w_over_omega0")
                mesh_rows.append({
                    "case_id": case_id,
                    "huang_rhys": s_value,
                    "j_sign": sign,
                    "j_meV": _float(row, "j_meV"),
                    "r_abs": r_abs,
                    "nu0_meV": nu0,
                    "g_meV": nu0 * math.sqrt(s_value),
                    "lambda_meV": s_value * nu0,
                    "eta_meV": ETA_OVER_NU * nu0,
                    "eta_over_nu0": _float(row, "eta_over_omega0"),
                    "w_W1_over_nu0": w_value,
                    "local_baseline_w": baseline_w,
                    "finite_J_delta_w": w_value - baseline_w,
                })
    if set(source_mesh_by_id) != {row["case_id"] for row in mesh_rows}:
        raise ValueError("Active line-source cases differ from the selected mesh")
    mesh_rows.sort(key=lambda row: (
        int(row["j_sign"]), float(row["huang_rhys"]), float(row["r_abs"])
    ))

    baseline_rows: List[Dict[str, Any]] = []
    for s_value in ALL_S:
        row = baseline_by_s[round(s_value, 12)]
        if (_float(row, "ed_raw_area") <= 0.0
                or _float(row, "scba_raw_area") <= 0.0
                or abs(_float(row, "scba_raw_area_relative_difference")) > 3.0e-6):
            raise ValueError("Local baseline raw-area check failed for S=%g" % s_value)
        baseline_rows.append({
            "case_id": row["case_id"],
            "huang_rhys": s_value,
            "nu0_meV": _float(row, "omega0_meV"),
            "eta_over_nu0": _float(row, "eta_over_omega0"),
            "w0_W1_over_nu0": _float(row, "scba_w_over_omega0"),
            "ed_raw_area": _float(row, "ed_raw_area"),
            "scba_raw_area": _float(row, "scba_raw_area"),
            "scba_raw_area_relative_difference": _float(
                row, "scba_raw_area_relative_difference"
            ),
        })

    extrema_source = _read_csv(sources["extrema"])
    if len(extrema_source) != 30:
        raise ValueError("Expected 30 selected-S/sign/guide extrema rows")
    extrema_rows: List[Dict[str, Any]] = []
    mesh_groups: Dict[Tuple[float, int], List[Dict[str, Any]]] = {}
    for s_value in SELECTED_S:
        for sign in (-1, +1):
            mesh_groups[(s_value, sign)] = sorted(
                (
                    row for row in mesh_rows
                    if _close(float(row["huang_rhys"]), s_value)
                    and int(row["j_sign"]) == sign
                ),
                key=lambda row: float(row["r_abs"]),
            )
    for source in extrema_source:
        s_value = _float(source, "huang_rhys")
        sign = int(round(_float(source, "j_sign")))
        guide = int(round(_float(source, "integer_guide")))
        rows = mesh_groups.get((s_value, sign))
        if rows is None or guide not in range(1, 6):
            raise ValueError("Unexpected extrema group")
        positions = np.asarray([float(row["r_abs"]) for row in rows])
        values = np.asarray([float(row["w_W1_over_nu0"]) for row in rows])
        core_indices = np.where(np.abs(positions - guide) <= 0.15 + 1.0e-12)[0]
        reference_indices = np.where(
            (np.abs(positions - guide) > 0.15 + 1.0e-12)
            & (np.abs(positions - guide) <= 0.50 + 1.0e-12)
        )[0]
        local_indices = np.asarray([
            index for index in range(1, len(rows) - 1)
            if abs(positions[index] - guide) < 0.50 - 1.0e-12
            and values[index] > values[index - 1]
            and values[index] > values[index + 1]
        ], dtype=int)
        if core_indices.size == 0 or reference_indices.size == 0:
            raise ValueError("Incomplete extrema window")
        if local_indices.size:
            peak_index = int(local_indices[np.argmax(values[local_indices])])
            extremum_type = "resolved local maximum"
        else:
            peak_index = int(core_indices[np.argmax(values[core_indices])])
            extremum_type = "guide-window maximum; no resolved local cusp"
        reference_index = int(reference_indices[np.argmin(values[reference_indices])])
        expected = {
            "r_peak": positions[peak_index],
            "w_peak": values[peak_index],
            "peak_case_id": rows[peak_index]["case_id"],
            "r_reference_min": positions[reference_index],
            "w_reference_min": values[reference_index],
            "reference_case_id": rows[reference_index]["case_id"],
        }
        for key in ("r_peak", "w_peak", "r_reference_min", "w_reference_min"):
            if not _close(_float(source, key), float(expected[key]), 2.0e-12):
                raise ValueError("Measured-extremum mismatch for %s" % key)
        for key in ("peak_case_id", "reference_case_id"):
            if str(source.get(key, "")) != str(expected[key]):
                raise ValueError("Measured-extremum case mismatch for %s" % key)
        if source.get("extremum_type") != extremum_type:
            raise ValueError("Measured-extremum status mismatch")
        output = dict(source)
        output["verified_against_measured_mesh"] = True
        extrema_rows.append(output)
    extrema_rows.sort(key=lambda row: (
        _float(row, "huang_rhys"), int(round(_float(row, "j_sign"))),
        int(round(_float(row, "integer_guide"))),
    ))
    negative_resolved = [
        row for row in extrema_rows
        if int(round(_float(row, "j_sign"))) < 0
        and row.get("extremum_type") == "resolved local maximum"
    ]
    if negative_resolved:
        raise ValueError("Negative-J guide windows contain a promoted cusp")

    cuts_source = _read_csv(sources["cuts"])
    if len(cuts_source) != 8:
        raise ValueError("Expected eight first-resonance spectral cuts")
    extrema_lookup = {
        (_float(row, "huang_rhys"), int(round(_float(row, "j_sign"))),
         int(round(_float(row, "integer_guide")))): row
        for row in extrema_rows
    }
    cut_rows: List[Dict[str, Any]] = []
    expected_order = 0
    for s_index, s_value in enumerate(SWEEP_S):
        group = sorted(
            (row for row in cuts_source if _close(_float(row, "huang_rhys"), s_value)),
            key=lambda row: int(round(_float(row, "display_order"))),
        )
        if len(group) != 4:
            raise ValueError("Expected four cuts for S=%g" % s_value)
        peak = extrema_lookup[(s_value, +1, 1)]
        expected_r = (
            0.65, _float(peak, "r_peak"), 1.0,
            _float(peak, "r_reference_min"),
        )
        if not all(expected_r[index] < expected_r[index + 1]
                   for index in range(3)):
            raise ValueError("The physical A--D cuts are not ordered")
        for cut_index, (source, r_expected) in enumerate(zip(group, expected_r)):
            expected_order += 1
            if (int(round(_float(source, "display_order"))) != expected_order
                    or source.get("cut_label") != CUT_LABELS[cut_index]
                    or int(round(_float(source, "j_sign"))) != +1
                    or not _close(_float(source, "r_abs"), r_expected)):
                raise ValueError("A--D cut definition changed for S=%g" % s_value)
            case_id = str(source.get("case_id", ""))
            summary_row = summary_by_id.get(case_id)
            if summary_row is None:
                raise KeyError("Cut case is absent from summary: %s" % case_id)
            archive_text = str(source.get("archive_file", ""))
            archive_path = bath_dir / archive_text
            if not archive_path.is_file():
                raise FileNotFoundError("Cut archive is missing: %s" % archive_path)
            required = (
                "energy_eV", "analysis_mask", "ed_green_bright_per_eV",
                "scba_green_bright_per_eV", "metadata_json",
            )
            with np.load(str(archive_path), allow_pickle=False) as archive:
                missing = [key for key in required if key not in archive.files]
                if missing:
                    raise KeyError("Archive %s lacks %s" % (archive_path, missing))
                energy = np.asarray(archive["energy_eV"], dtype=float)
                mask = np.asarray(archive["analysis_mask"], dtype=bool)
                ed_green = np.asarray(
                    archive["ed_green_bright_per_eV"], dtype=complex
                )
                scba_green = np.asarray(
                    archive["scba_green_bright_per_eV"], dtype=complex
                )
                metadata = json.loads(str(np.asarray(
                    archive["metadata_json"]
                ).item()))
            if metadata.get("case_id") != case_id:
                raise ValueError("Archive case-ID mismatch: %s" % archive_path)
            if (energy.ndim != 1 or mask.shape != energy.shape
                    or ed_green.shape != energy.shape
                    or scba_green.shape != energy.shape
                    or not np.all(np.diff(energy) > 0.0)
                    or not np.all(np.isfinite(ed_green))
                    or not np.all(np.isfinite(scba_green))):
                raise ValueError("Malformed cut archive: %s" % archive_path)
            selected_energy = energy[mask]
            ed_spectrum = -np.imag(ed_green[mask])
            scba_spectrum = -np.imag(scba_green[mask])
            minimum_ed = float(np.min(ed_spectrum))
            minimum_scba = float(np.min(scba_spectrum))
            if minimum_ed < -1.0e-10 or minimum_scba < -1.0e-10:
                raise ValueError("A cut archive violates spectral positivity")
            distance_eV, ed_area, scba_area = _continuous_wasserstein(
                selected_energy, ed_spectrum, scba_spectrum
            )
            nu0_meV = _float(summary_row, "omega0_meV")
            w_recomputed = distance_eV / (nu0_meV / 1000.0)
            w_stored = _float(summary_row, "scba_w_over_omega0")
            if abs(w_recomputed - w_stored) > 2.0e-9:
                raise ValueError("Archived Wasserstein mismatch for %s" % case_id)
            stored_ed_area = _float(summary_row, "ed_raw_area")
            stored_scba_area = _float(summary_row, "scba_raw_area")
            if (abs(ed_area / stored_ed_area - 1.0) > 5.0e-8
                    or abs(scba_area / stored_scba_area - 1.0) > 5.0e-8):
                raise ValueError("Archived/stored raw areas disagree for %s" % case_id)
            model = metadata.get("model", {})
            if (not _close(float(model.get("omega0_meV", float("nan"))), nu0_meV)
                    or not _close(float(model.get("huang_rhys", float("nan"))), s_value)
                    or not _close(float(model.get("j_meV", float("nan"))), J_ABS_MEV)
                    or not _close(
                        float(model.get("eta_meV", float("nan"))) / nu0_meV,
                        ETA_OVER_NU,
                    )):
                raise ValueError("Archived model metadata changed for %s" % case_id)
            cut_rows.append({
                "display_order": expected_order,
                "figure_panel": chr(ord("c") + s_index),
                "cut_label": CUT_LABELS[cut_index],
                "role": source.get("role", ""),
                "selection_rule": source.get("selection_rule", ""),
                "case_id": case_id,
                "huang_rhys": s_value,
                "j_sign": +1,
                "j_meV": J_ABS_MEV,
                "r_abs": _float(summary_row, "r_abs"),
                "nu0_meV": nu0_meV,
                "g_meV": nu0_meV * math.sqrt(s_value),
                "lambda_meV": nu0_meV * s_value,
                "eta_over_nu0": _float(summary_row, "eta_over_omega0"),
                "w_stored": w_stored,
                "w_recomputed_from_archive": w_recomputed,
                "w_absolute_difference": abs(w_recomputed - w_stored),
                "local_baseline_w": _float(source, "local_baseline_w_over_omega0"),
                "ed_raw_area_stored": stored_ed_area,
                "ed_raw_area_archive": ed_area,
                "scba_raw_area_stored": stored_scba_area,
                "scba_raw_area_archive": scba_area,
                "archive_area_relative_difference": scba_area / ed_area - 1.0,
                "minimum_ed_spectrum": minimum_ed,
                "minimum_scba_spectrum": minimum_scba,
                "archive_file": _project_relative(archive_path, project_root),
                "archive_sha256": _sha256(archive_path),
                "_energy_eV": energy,
                "_analysis_mask": mask,
                "_ed_spectrum": -np.imag(ed_green),
                "_scba_spectrum": -np.imag(scba_green),
                "_metadata": metadata,
            })

    return (
        mesh_rows, baseline_rows, extrema_rows, cut_rows,
        global_checks, sources,
    )


def load_revision27_refined_points(
    baseline_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Load independently calculated midpoint/refined-r publication points.

    These rows live outside the frozen 2,214-point production profile.  They
    use the same physical and numerical settings (Delta x=0.002, N_vib=10,
    eta/nu0=0.005) and are merged only for drawing the revision-27 lines.
    """

    audit_dir = SCRIPT_DIR / "revision27" / "data"
    source = audit_dir / "refined_r_points.csv"
    summary_path = audit_dir / "summary.json"
    if not source.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "Run revision27/audit_dimer_revision27.py before plotting"
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("schema_version") != "revision27_dimer_focused_audit_v1":
        raise ValueError("Unexpected revision-27 dimer audit schema")
    if bool(summary.get("publication_data_modified")):
        raise ValueError("Revision-27 audit must not modify production data")
    baseline_by_s = {
        round(_float(row, "huang_rhys"), 12): _float(row, "w0_W1_over_nu0")
        for row in baseline_rows
    }
    outputs: List[Dict[str, Any]] = []
    keys = set()
    for row in _read_csv(source):
        s_value = _float(row, "huang_rhys")
        sign = int(round(_float(row, "j_sign")))
        r_value = _float(row, "r_abs")
        dx = _float(row, "dx")
        phase = _float(row, "grid_phase_in_cells")
        n_vib = int(round(_float(row, "n_vib")))
        w_value = _float(row, "w_W1_over_nu0")
        residual = abs(_float(row, "scba_abs_residual_eV"))
        causality = _float(row, "scba_max_positive_im_sigma_eV")
        if (
            s_value not in (0.05, 0.10)
            or sign not in (-1, +1)
            or not _close(dx, PRODUCTION_DX, 2.0e-15)
            or not _close(phase, 0.0, 2.0e-15)
            or n_vib != 10
            or residual > 1.0e-9
            or causality > 5.0e-10
            or not np.isfinite(w_value)
        ):
            raise ValueError("A revision-27 refined point failed validation")
        key = (round(s_value, 12), sign, round(r_value, 12))
        if key in keys:
            raise ValueError("Duplicate revision-27 refined point")
        keys.add(key)
        nu0 = 2.0 * J_ABS_MEV / r_value
        baseline = baseline_by_s[round(s_value, 12)]
        outputs.append({
            "case_id": str(row["label"]),
            "huang_rhys": s_value,
            "j_sign": sign,
            "j_meV": sign * J_ABS_MEV,
            "r_abs": r_value,
            "nu0_meV": nu0,
            "g_meV": nu0 * math.sqrt(s_value),
            "lambda_meV": nu0 * s_value,
            "eta_meV": ETA_OVER_NU * nu0,
            "eta_over_nu0": ETA_OVER_NU,
            "w_W1_over_nu0": w_value,
            "local_baseline_w": baseline,
            "finite_J_delta_w": w_value - baseline,
            "revision27_refined": True,
        })
    outputs.sort(key=lambda row: (
        float(row["huang_rhys"]), int(row["j_sign"]), float(row["r_abs"])
    ))
    return outputs


def merge_calculated_mesh(
    production_rows: Sequence[Mapping[str, Any]],
    refined_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge calculated rows without interpolation or duplicate abscissae."""

    outputs = [dict(row) for row in production_rows]
    keys = {
        (round(_float(row, "huang_rhys"), 12),
         int(round(_float(row, "j_sign"))),
         round(_float(row, "r_abs"), 12))
        for row in outputs
    }
    for row in refined_rows:
        key = (
            round(_float(row, "huang_rhys"), 12),
            int(round(_float(row, "j_sign"))),
            round(_float(row, "r_abs"), 12),
        )
        if key in keys:
            raise ValueError("A refined r point duplicates the production mesh")
        keys.add(key)
        outputs.append(dict(row))
    outputs.sort(key=lambda row: (
        _float(row, "huang_rhys"), int(round(_float(row, "j_sign"))),
        _float(row, "r_abs"),
    ))
    return outputs


def _save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        str(path), format="pdf", bbox_inches="tight", pad_inches=0.025,
        metadata={
            "Title": path.stem,
            "Author": "single-mode dimer validation workflow",
            "Subject": "Calculated mesh points; no smoothing or fitting",
            "Creator": "Figures/common/dimer/single_mode/workflow.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def plot_error_scan(
    mesh_rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(7.20, 3.30), sharex=True, sharey=True
    )
    baseline_by_s = {
        round(float(row["huang_rhys"]), 12): float(row["w0_W1_over_nu0"])
        for row in baseline_rows
    }
    maximum = 0.0
    for panel_index, (axis, sign) in enumerate(zip(axes, SIGNS)):
        for s_value in SELECTED_S:
            group = sorted(
                (
                    row for row in mesh_rows
                    if _close(float(row["huang_rhys"]), s_value)
                    and int(row["j_sign"]) == sign
                ),
                key=lambda row: float(row["r_abs"]),
            )
            r_values = np.asarray([float(row["r_abs"]) for row in group])
            w_values = np.asarray([float(row["w_W1_over_nu0"]) for row in group])
            maximum = max(maximum, float(np.max(w_values)))
            axis.plot(
                r_values, w_values, color=S_COLORS[s_value],
                linewidth=0.95,
                label=r"$S=%g$" % s_value, zorder=5,
            )
            baseline = baseline_by_s[round(s_value, 12)]
            maximum = max(maximum, baseline)
            axis.axhline(
                baseline, color=S_COLORS[s_value], linestyle=BASELINE_DASH,
                linewidth=0.72, alpha=0.82, zorder=2,
            )
        axis.set_xlim(0.25, 5.5)
        axis.set_xticks((0.5, 1, 2, 3, 4, 5, 5.5))
        axis.set_xlabel(r"$r=2|J|/\nu_0$")
        style_axis(axis)
        _panel_label(axis, chr(ord("a") + panel_index))
    axes[0].set_ylabel(r"$w=W_1/\nu_0$")
    axes[0].set_ylim(0.0, max(0.0535, 1.055 * maximum))

    inset = axes[1].inset_axes([0.535, 0.50, 0.425, 0.425])
    baseline_s = np.asarray([float(row["huang_rhys"]) for row in baseline_rows])
    baseline_w = np.asarray([float(row["w0_W1_over_nu0"]) for row in baseline_rows])
    inset.plot(
        baseline_s, baseline_w, color="#666666", marker="o",
        markersize=2.5, linewidth=0.75,
    )
    for s_value in SELECTED_S:
        index = int(np.argmin(np.abs(baseline_s - s_value)))
        inset.scatter(
            [baseline_s[index]], [baseline_w[index]],
            color=S_COLORS[s_value], s=12.0, zorder=5,
        )
    inset.set_xscale("log")
    inset.set_yscale("log")
    inset.set_xlabel(r"$S$", fontsize=7.0, labelpad=0.4)
    inset.set_ylabel(r"$w(J=0)$", fontsize=7.0, labelpad=0.4)
    inset.tick_params(labelsize=6.4, pad=1.0)
    style_axis(inset, top=False, right=False, minor=True)

    handles = [
        Line2D([0], [0], color=S_COLORS[s_value], linewidth=1.35,
               label=r"$S=%g$" % s_value)
        for s_value in SELECTED_S
    ] + [
        Line2D([0], [0], color="#666666", linestyle=BASELINE_DASH,
               linewidth=0.80, label=r"$J=0$"),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=4, **compact_legend_kwargs(),
    )
    fig.subplots_adjust(
        left=0.088, right=0.992, bottom=0.215, top=0.985, wspace=0.075
    )
    _save_pdf(fig, output)


def plot_spectral_cuts(
    mesh_rows: Sequence[Dict[str, Any]],
    cut_rows: Sequence[Dict[str, Any]],
    output: Path,
) -> None:
    fig = plt.figure(figsize=(7.20, 5.65))
    grid = fig.add_gridspec(
        2, 2, height_ratios=(0.92, 2.05), hspace=0.22, wspace=0.22
    )
    scan_axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    spectrum_axes = [fig.add_subplot(grid[1, index]) for index in range(2)]

    for s_index, (axis, s_value) in enumerate(zip(scan_axes, SWEEP_S)):
        trace = sorted(
            (
                row for row in mesh_rows
                if _close(float(row["huang_rhys"]), s_value)
                and int(row["j_sign"]) == +1
                and SWEEP_R_WINDOW[0] - 1.0e-12
                <= float(row["r_abs"])
                <= SWEEP_R_WINDOW[1] + 1.0e-12
            ),
            key=lambda row: float(row["r_abs"]),
        )
        cuts = [
            row for row in cut_rows
            if _close(float(row["huang_rhys"]), s_value)
        ]
        r_values = np.asarray([float(row["r_abs"]) for row in trace])
        w_values = np.asarray([float(row["w_W1_over_nu0"]) for row in trace])
        baseline = float(cuts[0]["local_baseline_w"])
        axis.plot(
            r_values, w_values, color=S_COLORS[s_value],
            linewidth=0.95, zorder=4,
        )
        axis.axhline(
            baseline, color=S_COLORS[s_value], linestyle=BASELINE_DASH,
            linewidth=0.72, alpha=0.82, zorder=2,
        )
        axis.axvline(
            1.0, color="#9A9A9A", linestyle=(0, (1.4, 2.0)),
            linewidth=0.55, alpha=0.75, zorder=1,
        )
        for cut_index, row in enumerate(cuts):
            r_cut = float(row["r_abs"])
            w_cut = float(row["w_stored"])
            axis.scatter(
                [r_cut], [w_cut], marker="o", s=18.0,
                facecolor="white", edgecolor="#222222",
                linewidth=0.65, zorder=8,
            )
            axis.annotate(
                str(row["cut_label"]), (r_cut, w_cut), xytext=(0, 4),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7.4, fontweight="bold", color="black",
                zorder=9,
            )
        lower = min(float(np.min(w_values)), baseline)
        upper = max(float(np.max(w_values)), baseline)
        padding = max(0.11 * (upper - lower), 2.5e-4)
        axis.set_xlim(*SWEEP_R_WINDOW)
        axis.set_ylim(max(0.0, lower - padding), upper + 1.55 * padding)
        axis.set_xlabel(r"$r=2J/\nu_0$")
        axis.set_ylabel(r"$w=W_1/\nu_0$")
        axis.text(
            0.97, 0.94, r"$S=%g$" % s_value, transform=axis.transAxes,
            ha="right", va="top", fontsize=8.7,
        )
        style_axis(axis)
        _panel_label(axis, chr(ord("a") + s_index))

    spectral_handles: Optional[Tuple[Line2D, Line2D]] = None
    for s_index, (axis, s_value) in enumerate(zip(spectrum_axes, SWEEP_S)):
        cuts = [
            row for row in cut_rows
            if _close(float(row["huang_rhys"]), s_value)
        ]
        prepared: List[Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
        maximum = 0.0
        for row in cuts:
            energy = np.asarray(row["_energy_eV"], dtype=float)
            mask = np.asarray(row["_analysis_mask"], dtype=bool)
            ed_spectrum = np.asarray(row["_ed_spectrum"], dtype=float)
            scba_spectrum = np.asarray(row["_scba_spectrum"], dtype=float)
            nu0_eV = float(row["nu0_meV"]) / 1000.0
            metadata = row["_metadata"]
            model = metadata["model"]
            bright_energy = (
                float(model["energy_00_eV"])
                + float(model["huang_rhys"]) * nu0_eV
                + float(model["j_meV"]) / 1000.0
            )
            detuning = (energy - bright_energy) / nu0_eV
            display = (
                mask
                & (detuning >= SPECTRAL_X_WINDOW[0])
                & (detuning <= SPECTRAL_X_WINDOW[1])
            )
            ed_area = float(row["ed_raw_area_archive"])
            scba_area = float(row["scba_raw_area_archive"])
            ed_dimensionless = nu0_eV * ed_spectrum / ed_area
            scba_dimensionless = nu0_eV * scba_spectrum / scba_area
            maximum = max(
                maximum,
                float(np.max(ed_dimensionless[display])),
                float(np.max(scba_dimensionless[display])),
            )
            prepared.append((
                row, detuning[display], ed_dimensionless[display],
                scba_dimensionless[display],
            ))
        spacing = 1.20 * maximum
        offsets = np.arange(4, dtype=float) * spacing
        for cut_index, ((row, detuning, ed_values, scba_values), offset) in enumerate(
            zip(prepared, offsets)
        ):
            line_ed, = axis.plot(
                detuning, offset + ed_values, label="ED", **EXACT_STYLE,
            )
            line_scba, = axis.plot(
                detuning, offset + scba_values, label="SCBA", **SCBA_STYLE,
            )
            spectral_handles = (line_ed, line_scba)
            axis.axhline(offset, color="#C8C8C8", linewidth=0.38, zorder=1)
            # Put the physical cut labels inside each panel so the two
            # columns remain independent and no long y-tick text crosses the
            # central gutter.
            axis.text(
                SPECTRAL_X_WINDOW[0] + 0.035,
                offset + 0.035 * maximum,
                "%s" % row["cut_label"],
                ha="left", va="bottom", fontsize=7.2,
                fontweight="bold", color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.5),
                zorder=8,
            )
        axis.axvline(
            0.0, color="#A0A0A0", linestyle=(0, (1.4, 2.0)),
            linewidth=0.48, zorder=1,
        )
        axis.set_xlim(*SPECTRAL_X_WINDOW)
        axis.set_ylim(-0.07 * spacing, offsets[-1] + 1.08 * maximum)
        axis.set_yticks([])
        axis.set_xlabel(r"$(E-E_B)/\nu_0$")
        style_axis(axis)
        _panel_label(axis, chr(ord("c") + s_index))
    spectrum_axes[0].set_ylabel("Normalized intensity (offset)")
    if spectral_handles is None:
        raise RuntimeError("No spectral curves were plotted")
    fig.legend(
        handles=spectral_handles, loc="lower center", bbox_to_anchor=(0.5, 0.006),
        ncol=2, **compact_legend_kwargs(),
    )
    fig.subplots_adjust(left=0.102, right=0.992, bottom=0.075, top=0.985)
    _save_pdf(fig, output)


def configure_style() -> None:
    """Apply the common SI Matplotlib style."""

    _configure_style()


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    """Write a deterministic CSV atomically."""

    _write_csv_atomic(path, rows, columns)


def write_text(path: Path, text: str) -> None:
    """Write a text validation report atomically."""

    _write_text_atomic(path, text)


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""

    return _sha256(path)


def project_relative(path: Path, project_root: Path) -> str:
    """Return a normalized project-relative path."""

    return _project_relative(path, project_root)
