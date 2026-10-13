#!/usr/bin/env python3
"""Analyze exact, BA, and SCBA single-mode Frenkel-dimer spectra.

The active cases are read from ``scan_manifest.csv`` by default.  Each
completed case must point to one combined NPZ archive containing the ED, BA,
and SCBA Green functions and the resolved SCBA self-energy channels.  All
metrics are evaluated on the exact finite analysis window stored in the
archive metadata; manifest values never override those bounds.

This script is independent of the process working directory.  Relative CLI
paths are resolved below ``Figures/common/dimer/single_mode/source`` and relative
archive paths in a manifest are first resolved relative to that manifest.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parents[4]
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from single_mode_common import RESULTS  # noqa: E402
from single_mode_metrics import (  # noqa: E402
    PeakAnalysis,
    ProjectedSelfEnergyAtPeak,
    analyze_peaks,
    compare_peak_analyses,
    compare_spectra,
    dark_window_weight,
    flat_dict,
    match_sidebands,
    projected_self_energy_at_peak,
    raw_area,
    validate_uniform_grid,
)


DEFAULT_MANIFEST = RESULTS / "scan_manifest.csv"
DEFAULT_SUMMARY = RESULTS / "ed_scba_error_summary.csv"
DEFAULT_SIDEBANDS = RESULTS / "sideband_matching.csv"

GREEN_CHANNELS = ("bright", "dark", "local")
APPROXIMATIONS = ("ba", "scba")
SELF_ENERGY_CHANNELS = (
    "bright_emission",
    "bright_absorption",
    "dark_emission",
    "dark_absorption",
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
    "metadata_json",
)

TRUE_TEXT = {"1", "true", "yes", "y", "on", "active", "enabled"}
FALSE_TEXT = {"0", "false", "no", "n", "off", "inactive", "disabled"}
COMPLETE_TEXT = {
    "complete",
    "completed",
    "done",
    "success",
    "succeeded",
    "pass",
    "passed",
}
INCOMPLETE_TEXT = {
    "planned",
    "pending",
    "queued",
    "running",
    "incomplete",
    "failed",
    "error",
    "skipped",
}
ARCHIVE_COLUMNS = (
    "combined_npz",
    "combined_file",
    "combined_archive",
    "combined_archive_path",
    "archive",
    "archive_path",
    "result_npz",
    "results_npz",
    "output_npz",
    "output_file",
    "result_file",
)


def _project_relative(path: Path) -> str:
    """Return a stable project-relative path for persisted provenance."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError as error:
        raise ValueError(
            "Archive is outside the project root and cannot be recorded "
            "portably: {}".format(path)
        ) from error


def _normalized_key(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _mapping_value(
    mapping: Mapping[str, Any],
    aliases: Sequence[str],
) -> Optional[Any]:
    normalized = {_normalized_key(str(key)): value for key, value in mapping.items()}
    for alias in aliases:
        key = _normalized_key(alias)
        if key in normalized and str(normalized[key]).strip() != "":
            return normalized[key]
    return None


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in TRUE_TEXT:
        return True
    if text in FALSE_TEXT:
        return False
    raise ValueError("{} has an invalid Boolean value {!r}".format(name, value))


def _row_is_active(row: Mapping[str, Any]) -> bool:
    value = _mapping_value(row, ("active", "enabled", "run_case"))
    return True if value is None else _parse_bool(value, "manifest active flag")


def _row_has_explicit_active_flag(row: Mapping[str, Any]) -> bool:
    return _mapping_value(row, ("active", "enabled", "run_case")) is not None


def _row_is_complete(row: Mapping[str, Any]) -> Optional[bool]:
    completed = _mapping_value(
        row, ("completed", "complete", "is_complete", "run_completed")
    )
    if completed is not None:
        return _parse_bool(completed, "manifest completed flag")
    status = _mapping_value(row, ("status", "run_status", "state"))
    if status is None:
        return None
    text = str(status).strip().lower()
    if text in COMPLETE_TEXT:
        return True
    if text in INCOMPLETE_TEXT:
        return False
    raise ValueError("Unrecognized manifest status {!r}".format(status))


def _case_id(row: Mapping[str, Any], row_number: int) -> str:
    value = _mapping_value(row, ("case_id", "case", "id", "name"))
    if value is None:
        return "manifest_row_{:05d}".format(row_number)
    text = str(value).strip()
    if not text:
        raise ValueError("Manifest row {} has an empty case identifier".format(row_number))
    return text


def read_manifest(path: Path) -> List[Dict[str, str]]:
    """Read a nonempty CSV manifest while preserving all user columns."""

    if not path.is_file():
        raise FileNotFoundError("Scan manifest not found: {}".format(path))
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Scan manifest has no header: {}".format(path))
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("Scan manifest contains no cases: {}".format(path))
    return rows


def _archive_has_required_keys(path: Path) -> bool:
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            return all(key in archive.files for key in REQUIRED_ARCHIVE_KEYS)
    except (OSError, ValueError):
        return False


def resolve_archive_path(
    row: Mapping[str, Any],
    case_id: str,
    manifest_path: Path,
) -> Path:
    """Resolve an explicitly named or unambiguous derived combined archive."""

    value = _mapping_value(row, ARCHIVE_COLUMNS)
    if value is not None:
        specified = Path(str(value).strip()).expanduser()
        candidates = (
            [specified]
            if specified.is_absolute()
            else [manifest_path.parent / specified, MODULE_DIR / specified]
        )
        existing = [candidate.resolve() for candidate in candidates if candidate.is_file()]
        if not existing:
            raise FileNotFoundError(
                "Combined archive for {} does not exist: {}".format(
                    case_id, specified
                )
            )
        unique = list(dict.fromkeys(existing))
        if len(unique) > 1:
            raise RuntimeError(
                "Archive path for {} is ambiguous: {}".format(
                    case_id, ", ".join(str(path) for path in unique)
                )
            )
        return unique[0]

    direct = [
        RESULTS / "spectra" / (case_id + ".npz"),
        RESULTS / (case_id + ".npz"),
    ]
    candidates = [path.resolve() for path in direct if path.is_file()]
    if not candidates and RESULTS.is_dir():
        candidates = [
            path.resolve()
            for path in RESULTS.rglob("*.npz")
            if case_id in path.name
            and path.is_file()
            and _archive_has_required_keys(path)
        ]
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected one combined archive for {}, found {}. Add a manifest "
            "combined_npz column.".format(case_id, len(candidates))
        )
    return candidates[0]


def _decode_metadata(value: np.ndarray) -> Dict[str, Any]:
    scalar = np.asarray(value)
    if scalar.size != 1:
        raise ValueError("metadata_json must contain one JSON scalar")
    item = scalar.reshape(()).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    metadata = json.loads(str(item))
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json must decode to a JSON object")
    return metadata


def _collect_metadata_values(value: Any, alias: str) -> List[Any]:
    target = _normalized_key(alias)
    found: List[Any] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            if _normalized_key(str(key)) == target:
                found.append(nested)
            found.extend(_collect_metadata_values(nested, alias))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found.extend(_collect_metadata_values(nested, alias))
    return found


def _metadata_number(
    metadata: Mapping[str, Any],
    aliases: Sequence[str],
    required: bool = True,
) -> float:
    """Find one consistent numeric metadata value using ordered aliases."""

    for alias in aliases:
        raw_values = _collect_metadata_values(metadata, alias)
        if not raw_values:
            continue
        values: List[float] = []
        for raw in raw_values:
            try:
                number = float(raw)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                values.append(number)
        if not values:
            continue
        reference = values[0]
        if any(
            not np.isclose(value, reference, rtol=1.0e-10, atol=1.0e-12)
            for value in values[1:]
        ):
            raise ValueError(
                "Conflicting metadata values for {}: {}".format(alias, values)
            )
        return float(reference)
    if required:
        raise KeyError(
            "Metadata lacks required numeric field ({})".format(
                ", ".join(aliases)
            )
        )
    return float("nan")


def _row_number(row: Mapping[str, Any], aliases: Sequence[str]) -> float:
    value = _mapping_value(row, aliases)
    if value is None:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if np.isfinite(number) else float("nan")


def _metadata_or_row_number(
    metadata: Mapping[str, Any],
    row: Mapping[str, Any],
    aliases: Sequence[str],
) -> float:
    value = _metadata_number(metadata, aliases, required=False)
    return value if np.isfinite(value) else _row_number(row, aliases)


def _model_values(
    metadata: Mapping[str, Any],
    row: Mapping[str, Any],
) -> Dict[str, float]:
    omega0_eV = _metadata_or_row_number(
        metadata, row, ("omega0_eV", "phonon_frequency_eV")
    )
    if not np.isfinite(omega0_eV):
        omega0_meV = _metadata_or_row_number(
            metadata, row, ("omega0_meV", "omega_0_meV", "phonon_frequency_meV")
        )
        omega0_eV = omega0_meV * 1.0e-3
    eta_eV = _metadata_or_row_number(metadata, row, ("eta_eV",))
    if not np.isfinite(eta_eV):
        eta_eV = _metadata_or_row_number(metadata, row, ("eta_meV",)) * 1.0e-3
    j_eV = _metadata_or_row_number(metadata, row, ("j_eV", "coupling_J_eV"))
    if not np.isfinite(j_eV):
        j_eV = _metadata_or_row_number(metadata, row, ("j_meV", "J_meV")) * 1.0e-3
    huang_rhys = _metadata_or_row_number(
        metadata, row, ("huang_rhys", "S", "huang_rhys_factor")
    )
    energy_00_eV = _metadata_or_row_number(
        metadata, row, ("energy_00_eV", "E_00_eV", "E00_eV")
    )
    epsilon_v_eV = _metadata_or_row_number(
        metadata, row, ("epsilon_v_eV", "vertical_energy_eV")
    )
    lambda_eV = _metadata_or_row_number(
        metadata, row, ("lambda_eV", "reorganization_eV")
    )
    if not np.isfinite(lambda_eV) and np.isfinite(huang_rhys):
        lambda_eV = huang_rhys * omega0_eV
    if not np.isfinite(epsilon_v_eV) and np.isfinite(energy_00_eV):
        epsilon_v_eV = energy_00_eV + lambda_eV
    if not np.isfinite(energy_00_eV) and np.isfinite(epsilon_v_eV):
        energy_00_eV = epsilon_v_eV - lambda_eV
    temperature_K = _metadata_or_row_number(
        metadata, row, ("temperature_K", "T_K")
    )
    required = {
        "omega0_eV": omega0_eV,
        "eta_eV": eta_eV,
        "j_eV": j_eV,
        "epsilon_v_eV": epsilon_v_eV,
    }
    missing = [name for name, value in required.items() if not np.isfinite(value)]
    if missing:
        raise KeyError("Missing physical metadata: {}".format(", ".join(missing)))
    if omega0_eV <= 0.0 or eta_eV <= 0.0:
        raise ValueError("omega0 and eta must be positive")
    return {
        "energy_00_eV": float(energy_00_eV),
        "omega0_eV": float(omega0_eV),
        "omega0_meV": float(omega0_eV * 1.0e3),
        "huang_rhys": float(huang_rhys),
        "lambda_eV": float(lambda_eV),
        "lambda_meV": float(lambda_eV * 1.0e3),
        "j_eV": float(j_eV),
        "j_meV": float(j_eV * 1.0e3),
        "r_2J_over_omega0": float(2.0 * j_eV / omega0_eV),
        "temperature_K": float(temperature_K),
        "eta_eV": float(eta_eV),
        "eta_meV": float(eta_eV * 1.0e3),
        "epsilon_v_eV": float(epsilon_v_eV),
        "bare_bright_energy_eV": float(epsilon_v_eV + j_eV),
        "bare_dark_energy_eV": float(epsilon_v_eV - j_eV),
    }


def load_combined_archive(path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load and validate the required combined archive arrays."""

    if not path.is_file():
        raise FileNotFoundError("Combined archive not found: {}".format(path))
    with np.load(str(path), allow_pickle=False) as archive:
        missing = [key for key in REQUIRED_ARCHIVE_KEYS if key not in archive.files]
        if missing:
            raise KeyError(
                "{} lacks required arrays: {}".format(path, ", ".join(missing))
            )
        arrays = {
            key: np.asarray(archive[key]).copy()
            for key in REQUIRED_ARCHIVE_KEYS
            if key != "metadata_json"
        }
        metadata = _decode_metadata(archive["metadata_json"])

    energy, _ = validate_uniform_grid(arrays["energy_eV"])
    arrays["energy_eV"] = energy
    mask_raw = np.asarray(arrays["analysis_mask"])
    if mask_raw.shape != energy.shape:
        raise ValueError("analysis_mask and energy_eV shapes differ in {}".format(path))
    if not np.all((mask_raw == 0) | (mask_raw == 1)):
        raise ValueError("analysis_mask is not Boolean in {}".format(path))
    arrays["analysis_mask"] = mask_raw.astype(bool)

    for key, values in arrays.items():
        if key in ("energy_eV", "analysis_mask"):
            continue
        if values.ndim != 1 or values.shape != energy.shape:
            raise ValueError("{} has the wrong shape in {}".format(key, path))
        if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
            raise ValueError("{} contains NaN or Inf in {}".format(key, path))
    return arrays, metadata


def _analysis_window(
    metadata: Mapping[str, Any],
    energy: np.ndarray,
    saved_mask: np.ndarray,
) -> Tuple[float, float]:
    """Require metadata bounds and verify the serialized mask against them."""

    lower = _metadata_number(metadata, ("analysis_min_eV",))
    upper = _metadata_number(metadata, ("analysis_max_eV",))
    if not lower < upper:
        raise ValueError("Metadata analysis bounds are unordered")
    tolerance = 32.0 * np.finfo(float).eps * max(1.0, abs(lower), abs(upper))
    expected = (energy >= lower - tolerance) & (energy <= upper + tolerance)
    if np.count_nonzero(expected) < 3:
        raise ValueError("Metadata analysis window contains fewer than three samples")
    if not np.array_equal(expected, saved_mask):
        mismatch = int(np.count_nonzero(expected != saved_mask))
        raise ValueError(
            "Serialized analysis_mask disagrees with metadata at {} samples".format(
                mismatch
            )
        )
    return float(lower), float(upper)


def _complex_at(
    energy: np.ndarray,
    values: np.ndarray,
    target_eV: float,
) -> complex:
    if target_eV < energy[0] or target_eV > energy[-1]:
        raise ValueError("Interpolation target lies outside the archive grid")
    return complex(
        np.interp(target_eV, energy, values.real),
        np.interp(target_eV, energy, values.imag),
    )


def _manifest_columns(row: Mapping[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for key, value in row.items():
        safe = "".join(
            character if character.isalnum() else "_" for character in str(key)
        ).strip("_")
        values["manifest_" + safe] = value
    return values


def _peak_summary_columns(prefix: str, analysis: PeakAnalysis) -> Dict[str, Any]:
    return {
        prefix + "resolved_peak_count": len(analysis.peaks),
        prefix + "dominant_peak_ordinal": analysis.dominant_ordinal,
        prefix + "dominant_peak_energy_eV": analysis.dominant_energy_eV,
        prefix + "dominant_fwhm_eV": analysis.dominant_fwhm_eV,
        prefix + "dominant_fwhm_reliable": analysis.dominant_fwhm_reliable,
        prefix + "peak_ambiguity_reason": analysis.ambiguity_reason,
    }


def _self_energy_columns(
    metric: ProjectedSelfEnergyAtPeak,
) -> Dict[str, Any]:
    return flat_dict(metric, prefix="exact_projected_")


def _zero_temperature_common_mode_limit_green(
    energy_eV: np.ndarray,
    model: Mapping[str, float],
    tail_tolerance: float = 1.0e-14,
) -> np.ndarray:
    """Bright Green function after suppressing difference-mode conversion.

    The surviving common oscillator has Huang--Rhys factor S/2.  Its relaxed
    zero-phonon line is E00 + lambda/2 + J, rather than a phonon-free pole.
    """

    effective_s = 0.5 * float(model["huang_rhys"])
    weights = [float(np.exp(-effective_s))]
    cumulative = weights[0]
    sideband = 0
    while 1.0 - cumulative > tail_tolerance:
        sideband += 1
        weights.append(weights[-1] * effective_s / sideband)
        cumulative += weights[-1]
        if sideband > 10000:
            raise RuntimeError("Common-mode Franck--Condon tail did not converge")
    centers = (
        float(model["energy_00_eV"])
        + 0.5 * float(model["lambda_eV"])
        + float(model["j_eV"])
        + np.arange(len(weights), dtype=float) * float(model["omega0_eV"])
    )
    green = np.zeros_like(energy_eV, dtype=complex)
    eta = float(model["eta_eV"])
    for center, weight in zip(centers, weights):
        green += weight / (energy_eV - center + 1.0j * eta)
    return green


def analyze_archive(
    archive_path: Path,
    manifest_row: Mapping[str, Any],
    case_id: str,
    prominence_fraction: float = 0.01,
    matching_method: str = "monotone",
    negative_tolerance: float = 1.0e-10,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Analyze one combined ED/BA/SCBA archive."""

    arrays, metadata = load_combined_archive(archive_path)
    energy = arrays["energy_eV"]
    _, spacing_eV = validate_uniform_grid(energy)
    window = _analysis_window(metadata, energy, arrays["analysis_mask"])
    model = _model_values(metadata, manifest_row)
    eta_eV = model["eta_eV"]
    omega0_eV = model["omega0_eV"]
    minimum_peak_distance_eV = max(2.0 * eta_eV, spacing_eV)
    matching_radius_eV = max(4.0 * eta_eV, 0.35 * omega0_eV)
    dark_half_width_eV = max(4.0 * eta_eV, 0.25 * omega0_eV)

    greens: Dict[str, Dict[str, np.ndarray]] = {}
    spectra: Dict[str, Dict[str, np.ndarray]] = {}
    for method in ("ed", "ba", "scba"):
        greens[method] = {}
        spectra[method] = {}
        for channel in GREEN_CHANNELS:
            key = "{}_green_{}_per_eV".format(method, channel)
            green = np.asarray(arrays[key], dtype=complex)
            greens[method][channel] = green
            spectra[method][channel] = -np.imag(green)

    peak_analyses: Dict[str, PeakAnalysis] = {}
    for method in ("ed", "ba", "scba"):
        peak_analyses[method] = analyze_peaks(
            energy,
            spectra[method]["bright"],
            window=window,
            prominence_fraction=prominence_fraction,
            minimum_distance_eV=minimum_peak_distance_eV,
            negative_tolerance=negative_tolerance,
        )

    summary: Dict[str, Any] = {
        "case_id": case_id,
        "combined_archive": _project_relative(archive_path),
        **model,
        "analysis_min_eV": window[0],
        "analysis_max_eV": window[1],
        "analysis_grid_step_eV": spacing_eV,
        "analysis_grid_step_meV": spacing_eV * 1.0e3,
        "analysis_sample_count": int(np.count_nonzero(arrays["analysis_mask"])),
        "peak_prominence_fraction": prominence_fraction,
        "peak_minimum_distance_eV": minimum_peak_distance_eV,
        "peak_dominance_height_ratio": (
            peak_analyses["ed"].dominance_height_ratio_required
        ),
        "peak_dominance_weight_fraction": (
            peak_analyses["ed"].dominance_weight_fraction_required
        ),
        "sideband_matching_method": matching_method,
        "sideband_matching_radius_eV": matching_radius_eV,
        "dark_window_half_width_eV": dark_half_width_eV,
        **_manifest_columns(manifest_row),
    }

    # Raw auxiliary sum rules are reported for all saved observables.  Full
    # shape metrics below use the primary optical bright absorption only.
    for method in ("ed", "ba", "scba"):
        for channel in GREEN_CHANNELS:
            summary["{}_{}_raw_area".format(method, channel)] = raw_area(
                energy, spectra[method][channel], window
            )
    for approximation in APPROXIMATIONS:
        for channel in GREEN_CHANNELS:
            summary[
                "{}_{}_raw_area_error_candidate_minus_ed".format(
                    approximation, channel
                )
            ] = (
                summary["{}_{}_raw_area".format(approximation, channel)]
                - summary["ed_{}_raw_area".format(channel)]
            )

    summary.update(_peak_summary_columns("ed_", peak_analyses["ed"]))
    sideband_rows: List[Dict[str, Any]] = []
    for approximation in APPROXIMATIONS:
        comparison = compare_spectra(
            energy,
            spectra["ed"]["bright"],
            spectra[approximation]["bright"],
            window=window,
            negative_tolerance=negative_tolerance,
        )
        summary.update(flat_dict(comparison, prefix=approximation + "_"))
        summary.update(
            _peak_summary_columns(
                approximation + "_", peak_analyses[approximation]
            )
        )
        peak_comparison = compare_peak_analyses(
            peak_analyses["ed"], peak_analyses[approximation]
        )
        summary.update(
            flat_dict(
                peak_comparison,
                prefix=approximation + "_peak_comparison_",
            )
        )

        ed_dark_window = dark_window_weight(
            energy,
            spectra["ed"]["bright"],
            model["bare_dark_energy_eV"],
            dark_half_width_eV,
            analysis_window=window,
        )
        approximation_dark_window = dark_window_weight(
            energy,
            spectra[approximation]["bright"],
            model["bare_dark_energy_eV"],
            dark_half_width_eV,
            analysis_window=window,
        )
        summary["ed_bright_dark_window_raw_weight"] = ed_dark_window.raw_weight
        summary["ed_bright_dark_window_fraction"] = (
            ed_dark_window.fraction_of_analysis_area
        )
        summary[
            approximation + "_bright_dark_window_raw_weight"
        ] = approximation_dark_window.raw_weight
        summary[
            approximation + "_bright_dark_window_fraction"
        ] = approximation_dark_window.fraction_of_analysis_area
        summary[
            approximation + "_bright_dark_window_raw_weight_error"
        ] = approximation_dark_window.raw_weight - ed_dark_window.raw_weight
        summary[
            approximation + "_bright_dark_window_fraction_error"
        ] = (
            approximation_dark_window.fraction_of_analysis_area
            - ed_dark_window.fraction_of_analysis_area
        )

        matching = match_sidebands(
            peak_analyses["ed"].peaks,
            peak_analyses[approximation].peaks,
            maximum_energy_difference_eV=matching_radius_eV,
            method=matching_method,
        )
        aggregate = asdict(matching)
        aggregate.pop("rows")
        summary.update(
            {
                approximation + "_sideband_" + key: value
                for key, value in aggregate.items()
            }
        )
        for match_row in matching.rows:
            row_values: Dict[str, Any] = {
                "case_id": case_id,
                "combined_archive": _project_relative(archive_path),
                "approximation": approximation.upper(),
                "j_meV": model["j_meV"],
                "r_2J_over_omega0": model["r_2J_over_omega0"],
                "huang_rhys": model["huang_rhys"],
                "omega0_meV": model["omega0_meV"],
                "temperature_K": model["temperature_K"],
                "eta_meV": model["eta_meV"],
                "analysis_min_eV": window[0],
                "analysis_max_eV": window[1],
                "prominence_fraction": prominence_fraction,
                "minimum_peak_distance_eV": minimum_peak_distance_eV,
                "matching_method": matching.method,
                "matching_radius_eV": matching.maximum_energy_difference_eV,
                "matched_count": matching.matched_count,
                "missing_count": matching.missing_count,
                "spurious_count": matching.spurious_count,
                "matched_energy_mae_eV": matching.matched_energy_mae_eV,
                "matched_energy_rmse_eV": matching.matched_energy_rmse_eV,
                "aggregate_absolute_weight_error": (
                    matching.aggregate_absolute_weight_error
                ),
                "aggregate_relative_weight_error": (
                    matching.aggregate_relative_weight_error
                ),
            }
            row_values.update(asdict(match_row))
            sideband_rows.append(row_values)

    # H5 diagnostic: at T=0 compare exact ED with the phonon-dressed
    # large-splitting limit in which only the common S/2 mode remains.  This
    # comparator is meaningful for every T=0 row, while the large_splitting
    # manifest family supplies the actual asymptotic sequence in |r|.
    if abs(model["temperature_K"]) < 1.0e-12:
        common_green = _zero_temperature_common_mode_limit_green(energy, model)
        common_spectrum = -np.imag(common_green)
        common_comparison = compare_spectra(
            energy,
            common_spectrum,
            spectra["ed"]["bright"],
            window=window,
            negative_tolerance=negative_tolerance,
        )
        summary.update(
            flat_dict(common_comparison, prefix="common_mode_limit_vs_ed_")
        )
        common_peaks = analyze_peaks(
            energy,
            common_spectrum,
            window=window,
            prominence_fraction=prominence_fraction,
            minimum_distance_eV=minimum_peak_distance_eV,
            negative_tolerance=negative_tolerance,
        )
        summary.update(
            flat_dict(
                compare_peak_analyses(common_peaks, peak_analyses["ed"]),
                prefix="common_mode_limit_vs_ed_peak_",
            )
        )

    # The exact projected scalar self-energy is an operational diagnostic:
    # the true exact dimer self-energy is not assumed to be site-local.
    ed_peaks = peak_analyses["ed"]
    if ed_peaks.peaks:
        ed_peak_energy = ed_peaks.dominant_energy_eV
        exact_sigma = projected_self_energy_at_peak(
            energy,
            greens["ed"]["bright"],
            model["bare_bright_energy_eV"],
            eta_eV,
            ed_peak_energy,
        )
        summary.update(_self_energy_columns(exact_sigma))
        summary["exact_projected_sigma_real_eV_at_ed_peak"] = (
            exact_sigma.sigma_real_eV
        )
        summary["exact_projected_sigma_imag_eV_at_ed_peak"] = (
            exact_sigma.sigma_imag_eV
        )
        summary["exact_projected_minus_two_imag_sigma_eV_at_ed_peak"] = (
            exact_sigma.minus_two_imag_sigma_eV
        )
        scba_sigma = _complex_at(
            energy, np.asarray(arrays["scba_sigma_eV"], dtype=complex), ed_peak_energy
        )
        summary["scba_sigma_real_eV_at_ed_peak"] = scba_sigma.real
        summary["scba_sigma_imag_eV_at_ed_peak"] = scba_sigma.imag
        summary["scba_minus_two_imag_sigma_eV_at_ed_peak"] = -2.0 * scba_sigma.imag
        summary["scba_minus_exact_projected_re_sigma_eV_at_ed_peak"] = (
            scba_sigma.real - exact_sigma.sigma_real_eV
        )
        summary[
            "scba_minus_exact_projected_minus_two_im_sigma_eV_at_ed_peak"
        ] = -2.0 * scba_sigma.imag - exact_sigma.minus_two_imag_sigma_eV

        channel_sum = 0.0 + 0.0j
        for channel in SELF_ENERGY_CHANNELS:
            value = _complex_at(
                energy,
                np.asarray(
                    arrays["scba_sigma_from_{}_eV".format(channel)],
                    dtype=complex,
                ),
                ed_peak_energy,
            )
            channel_sum += value
            prefix = "scba_sigma_from_{}_at_ed_peak_".format(channel)
            summary[prefix + "real_eV"] = value.real
            summary[prefix + "imag_eV"] = value.imag
            summary[prefix + "minus_two_imag_eV"] = -2.0 * value.imag
        summary["scba_channel_sum_real_eV_at_ed_peak"] = channel_sum.real
        summary["scba_channel_sum_imag_eV_at_ed_peak"] = channel_sum.imag
        summary["scba_channel_closure_abs_error_eV_at_ed_peak"] = abs(
            channel_sum - scba_sigma
        )
    else:
        nan = float("nan")
        unavailable_exact = ProjectedSelfEnergyAtPeak(
            energy_eV=nan,
            bare_projected_energy_eV=model["bare_bright_energy_eV"],
            eta_eV=eta_eV,
            green_real_per_eV=nan,
            green_imag_per_eV=nan,
            sigma_real_eV=nan,
            sigma_imag_eV=nan,
            minus_two_imag_sigma_eV=nan,
        )
        summary.update(_self_energy_columns(unavailable_exact))
        summary["exact_projected_sigma_real_eV_at_ed_peak"] = nan
        summary["exact_projected_sigma_imag_eV_at_ed_peak"] = nan
        summary["exact_projected_minus_two_imag_sigma_eV_at_ed_peak"] = nan
        summary["scba_sigma_real_eV_at_ed_peak"] = float("nan")
        summary["scba_sigma_imag_eV_at_ed_peak"] = float("nan")
        summary["scba_minus_two_imag_sigma_eV_at_ed_peak"] = float("nan")
        summary["scba_minus_exact_projected_re_sigma_eV_at_ed_peak"] = nan
        summary[
            "scba_minus_exact_projected_minus_two_im_sigma_eV_at_ed_peak"
        ] = nan
        for channel in SELF_ENERGY_CHANNELS:
            prefix = "scba_sigma_from_{}_at_ed_peak_".format(channel)
            summary[prefix + "real_eV"] = nan
            summary[prefix + "imag_eV"] = nan
            summary[prefix + "minus_two_imag_eV"] = nan
        summary["scba_channel_sum_real_eV_at_ed_peak"] = nan
        summary["scba_channel_sum_imag_eV_at_ed_peak"] = nan
        summary["scba_channel_closure_abs_error_eV_at_ed_peak"] = nan
    summary["self_energy_at_ed_peak_available"] = bool(ed_peaks.peaks)
    return summary, sideband_rows


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fallback_fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for field in fallback_fields:
        if field not in seen:
            fields.append(field)
            seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def analyze_manifest(
    manifest_path: Path,
    summary_path: Path,
    sideband_path: Path,
    prominence_fraction: float = 0.01,
    matching_method: str = "monotone",
    negative_tolerance: float = 1.0e-10,
    allow_incomplete: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Analyze every active completed manifest case and write both CSVs."""

    manifest_rows = read_manifest(manifest_path)
    summaries: List[Dict[str, Any]] = []
    sidebands: List[Dict[str, Any]] = []
    case_ids = set()
    for row_number, row in enumerate(manifest_rows, start=2):
        if not _row_is_active(row):
            continue
        case_id = _case_id(row, row_number)
        if case_id in case_ids:
            raise ValueError("Duplicate active case_id in manifest: {}".format(case_id))
        case_ids.add(case_id)
        completion = _row_is_complete(row)
        if completion is False:
            if _row_has_explicit_active_flag(row) and not allow_incomplete:
                raise RuntimeError(
                    "Explicitly active case {} is not complete; use "
                    "--allow-incomplete to skip it".format(case_id)
                )
            # The production manifest intentionally contains pending cases
            # from scan families not run in the current stage.  With no
            # explicit active selector, status=complete defines the analyzed
            # subset and prevents stale/unwritten cases from contaminating it.
            continue
        try:
            archive_path = resolve_archive_path(row, case_id, manifest_path)
        except (FileNotFoundError, RuntimeError):
            if allow_incomplete:
                continue
            raise
        summary, rows = analyze_archive(
            archive_path,
            row,
            case_id,
            prominence_fraction=prominence_fraction,
            matching_method=matching_method,
            negative_tolerance=negative_tolerance,
        )
        summaries.append(summary)
        sidebands.extend(rows)

    if not summaries:
        raise RuntimeError("No active completed combined archives were analyzed")
    _write_csv(summary_path, summaries, ("case_id", "combined_archive"))
    _write_csv(
        sideband_path,
        sidebands,
        (
            "case_id",
            "combined_archive",
            "approximation",
            "status",
            "reference_ordinal",
            "candidate_ordinal",
            "reference_energy_eV",
            "candidate_energy_eV",
            "energy_shift_candidate_minus_reference_eV",
            "reference_weight",
            "candidate_weight",
            "weight_difference_candidate_minus_reference",
            "absolute_weight_error",
        ),
    )
    return summaries, sidebands


def _resolve_cli_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (MODULE_DIR / path).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="scan manifest CSV (relative paths resolve below this script)",
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--sidebands", type=Path, default=DEFAULT_SIDEBANDS)
    parser.add_argument("--prominence", type=float, default=0.01)
    parser.add_argument(
        "--matching-method",
        choices=("monotone", "hungarian"),
        default="monotone",
    )
    parser.add_argument("--negative-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="skip active manifest rows without completed combined archives",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if not 0.0 < args.prominence <= 1.0:
        raise ValueError("--prominence must lie in (0, 1]")
    if args.negative_tolerance < 0.0:
        raise ValueError("--negative-tolerance must be nonnegative")
    manifest = _resolve_cli_path(args.manifest)
    summary = _resolve_cli_path(args.summary)
    sidebands = _resolve_cli_path(args.sidebands)
    summary_rows, sideband_rows = analyze_manifest(
        manifest,
        summary,
        sidebands,
        prominence_fraction=args.prominence,
        matching_method=args.matching_method,
        negative_tolerance=args.negative_tolerance,
        allow_incomplete=args.allow_incomplete,
    )
    print("Analyzed {} completed cases".format(len(summary_rows)))
    print("Saved summary: {}".format(summary))
    print(
        "Saved {} sideband rows: {}".format(len(sideband_rows), sidebands)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
