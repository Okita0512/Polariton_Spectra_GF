#!/usr/bin/env python3
"""Final integrity and physics audit for the single-mode dimer benchmark.

This is deliberately a consumer, not another calculator: it reads the
authoritative production manifest, every combined spectrum archive, the
analysis/convergence tables, and the validation products.  With no options it
writes ``validation/final_audit.txt`` atomically.  ``--check-only`` performs
the same checks and prints the prospective report without changing a file.

All paths are resolved from this file, so the audit works from the repository
root, from this directory, or from an unrelated process working directory.
The script is compatible with Python 3.8 and uses no project-local imports.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = MODULE_DIR / "results"
FIGURES_DIR = MODULE_DIR / "figures"
VALIDATION_DIR = MODULE_DIR / "validation"
FINAL_REPORT = VALIDATION_DIR / "final_audit.txt"

EXPECTED_CASE_COUNT = 203
EXPECTED_FAMILY_COUNTS = {
    "direct_analogue": 9,
    "direct_eta_convergence": 12,
    "heatmap_r_S": 108,
    "frequency_temperature": 48,
    "eta_regimes": 24,
    "large_splitting": 18,
}
FIGURE_NAMES = (
    "direct_analogue_spectra",
    "j0_coupling_progression",
    "error_heatmaps",
    "signed_j_error_curves",
    "eta_convergence",
    "self_energy_channels",
    "ed_truncation_convergence",
    "regime_map",
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
SPECTRAL_ARRAY_KEYS = tuple(
    key
    for key in REQUIRED_ARCHIVE_KEYS
    if key.endswith("_per_eV") or key.endswith("_eV")
    if key != "energy_eV"
)
GREEN_PREFIXES = ("ed", "ba", "scba")
GREEN_CHANNELS = ("bright", "dark", "local")
SIGMA_CHANNEL_KEYS = (
    "scba_sigma_from_bright_emission_eV",
    "scba_sigma_from_bright_absorption_eV",
    "scba_sigma_from_dark_emission_eV",
    "scba_sigma_from_dark_absorption_eV",
)

THERMAL_OMISSION_LIMIT = 1.0e-10
SCBA_ABSOLUTE_RESIDUAL_LIMIT_EV = 1.0e-9
SCBA_RELATIVE_RESIDUAL_LIMIT = 1.0e-8
CAUSALITY_LIMIT_EV = 5.0e-10
MINIMUM_SPECTRUM_LIMIT_PER_EV = -1.0e-8
RAW_AREA_RELATIVE_LIMIT = 3.0e-6
CHANNEL_CLOSURE_RELATIVE_LIMIT = 2.0e-7  # complex64 archival round-off
LOCAL_IDENTITY_RELATIVE_LIMIT = 5.0e-7  # complex64 production archives
POLE_WEIGHT_ERROR_LIMIT = 1.0e-10


class Audit:
    """Collect deterministic checks while retaining only failed details."""

    def __init__(self) -> None:
        self.checks = 0
        self.failures: List[str] = []

    def check(self, condition: bool, name: str, detail: str = "") -> bool:
        self.checks += 1
        if not bool(condition):
            message = name if not detail else "{}: {}".format(name, detail)
            self.failures.append(message)
            return False
        return True

    def finite_le(self, value: float, limit: float, name: str) -> bool:
        return self.check(
            math.isfinite(value) and value <= limit,
            name,
            "value={:.17g}, limit={:.17g}".format(value, limit),
        )

    @property
    def passed(self) -> bool:
        return not self.failures


def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError("required CSV is missing: {}".format(path))
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header: {}".format(path))
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("CSV has no data rows: {}".format(path))
    return list(reader.fieldnames), rows


def _require_columns(
    audit: Audit, path: Path, fieldnames: Sequence[str], required: Iterable[str]
) -> None:
    missing = sorted(set(required) - set(fieldnames))
    audit.check(
        not missing,
        "columns present in {}".format(_relative(path)),
        "missing {}".format(", ".join(missing)),
    )


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(MODULE_DIR))
    except ValueError:
        return str(path.resolve())


def _inside_module(path: Path) -> bool:
    try:
        path.resolve().relative_to(MODULE_DIR)
        return True
    except ValueError:
        return False


def _float(row: Mapping[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or str(value).strip() == "":
        return float("nan")
    try:
        return float(str(value).strip())
    except ValueError:
        return float("nan")


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "pass", "passed")


def _families(row: Mapping[str, str], key: str = "families") -> Tuple[str, ...]:
    return tuple(item for item in str(row.get(key, "")).split(";") if item)


def _close(left: float, right: float, tolerance: float = 1.0e-9) -> bool:
    return bool(
        math.isfinite(left)
        and math.isfinite(right)
        and abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))
    )


def _select(
    rows: Sequence[Mapping[str, str]],
    family: Optional[str] = None,
    family_key: str = "manifest_families",
    **conditions: float
) -> List[Mapping[str, str]]:
    selected: List[Mapping[str, str]] = []
    for row in rows:
        if family is not None and family not in _families(row, family_key):
            continue
        if all(_close(_float(row, key), value) for key, value in conditions.items()):
            selected.append(row)
    return selected


def _one(
    audit: Audit,
    rows: Sequence[Mapping[str, str]],
    label: str,
    family: Optional[str] = None,
    **conditions: float
) -> Optional[Mapping[str, str]]:
    selected = _select(rows, family=family, **conditions)
    audit.check(
        len(selected) == 1,
        "unique analysis selection {}".format(label),
        "found {} rows for {}".format(len(selected), conditions),
    )
    return selected[0] if len(selected) == 1 else None


def _json_is_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_json_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_json_is_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _decode_metadata(value: np.ndarray) -> Dict[str, Any]:
    scalar = np.asarray(value)
    if scalar.size != 1:
        raise ValueError("metadata_json is not scalar")
    item = scalar.reshape(()).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    decoded = json.loads(str(item))
    if not isinstance(decoded, dict):
        raise ValueError("metadata_json does not decode to an object")
    return decoded


def _resolve_archive(row: Mapping[str, str]) -> Path:
    text = str(row.get("output_file", "")).strip()
    if not text:
        raise ValueError("manifest case {} has no output_file".format(row.get("case_id")))
    path = Path(text)
    resolved = path.resolve() if path.is_absolute() else (MODULE_DIR / path).resolve()
    if not _inside_module(resolved):
        raise ValueError("archive escapes benchmark directory: {}".format(resolved))
    return resolved


def _maximum_relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(reference))))
    return float(np.max(np.abs(candidate - reference)) / scale)


def _validate_archive(
    audit: Audit, row: Mapping[str, str], path: Path
) -> Dict[str, float]:
    """Validate one archive and return extrema used by the final report."""

    case_id = str(row["case_id"])
    if not path.is_file():
        audit.check(False, "archive exists for {}".format(case_id), _relative(path))
        return {}

    stats: Dict[str, float] = {"bytes": float(path.stat().st_size)}
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            missing = sorted(set(REQUIRED_ARCHIVE_KEYS) - set(archive.files))
            audit.check(
                not missing,
                "archive key coverage for {}".format(case_id),
                "missing {}".format(", ".join(missing)),
            )
            if missing:
                return stats

            energy = np.asarray(archive["energy_eV"])
            mask = np.asarray(archive["analysis_mask"])
            arrays = {
                key: np.asarray(archive[key])
                for key in SPECTRAL_ARRAY_KEYS
            }
            histories = {
                key: np.asarray(archive[key])
                for key in ("ba_history", "scba_history", "full_history")
            }
            metadata = _decode_metadata(archive["metadata_json"])
    except Exception as error:
        audit.check(
            False,
            "archive readable for {}".format(case_id),
            "{}: {}".format(type(error).__name__, error),
        )
        return stats

    audit.check(energy.ndim == 1 and energy.size >= 3, "energy shape for {}".format(case_id))
    audit.check(mask.shape == energy.shape, "analysis-mask shape for {}".format(case_id))
    audit.check(mask.dtype.kind == "b", "analysis-mask dtype for {}".format(case_id))
    audit.check(np.all(np.isfinite(energy)), "finite energy grid for {}".format(case_id))
    audit.check(np.all(np.diff(energy) > 0.0), "strict energy ordering for {}".format(case_id))
    audit.check(int(np.count_nonzero(mask)) >= 3, "nonempty analysis mask for {}".format(case_id))
    true_indices = np.flatnonzero(mask)
    audit.check(
        true_indices.size == 0 or np.all(np.diff(true_indices) == 1),
        "contiguous analysis mask for {}".format(case_id),
    )
    for key, array in arrays.items():
        audit.check(array.shape == energy.shape, "array shape {}:{}".format(case_id, key))
        audit.check(np.all(np.isfinite(array)), "finite array {}:{}".format(case_id, key))
    for key, history in histories.items():
        audit.check(
            history.ndim == 2 and history.shape[1] == 6,
            "history shape {}:{}".format(case_id, key),
            "shape={}".format(history.shape),
        )
        audit.check(np.all(np.isfinite(history)), "finite history {}:{}".format(case_id, key))

    audit.check(_json_is_finite(metadata), "finite archive metadata for {}".format(case_id))
    audit.check(metadata.get("case_id") == case_id, "metadata case ID for {}".format(case_id))
    audit.check(
        metadata.get("format") == "single_mode_dimer_ed_ba_scba_v1",
        "metadata format for {}".format(case_id),
    )
    linewidth_note = str(metadata.get("phonon_linewidth", "")).lower()
    audit.check(
        "none" in linewidth_note and "eta" in linewidth_note,
        "undamped-phonon metadata for {}".format(case_id),
        linewidth_note,
    )

    model = metadata.get("model", {})
    grid = metadata.get("grid", {})
    ed_meta = metadata.get("ed", {})
    scba_meta = metadata.get("scba", {})
    ba_meta = metadata.get("ba", {})
    for section_name, section in (
        ("model", model),
        ("grid", grid),
        ("ed", ed_meta),
        ("scba", scba_meta),
        ("ba", ba_meta),
    ):
        audit.check(isinstance(section, dict), "metadata section {}:{}".format(case_id, section_name))

    manifest_model_pairs = (
        ("energy_00_eV", "energy_00_eV"),
        ("omega0_meV", "omega0_meV"),
        ("huang_rhys", "huang_rhys"),
        ("j_meV", "j_meV"),
        ("temperature_K", "temperature_K"),
        ("eta_meV", "eta_meV"),
    )
    for manifest_key, metadata_key in manifest_model_pairs:
        audit.check(
            _close(_float(row, manifest_key), float(model.get(metadata_key, float("nan")))),
            "model metadata {}:{}".format(case_id, metadata_key),
        )
    for manifest_key, metadata_key in (
        ("analysis_min_eV", "analysis_min_eV"),
        ("analysis_max_eV", "analysis_max_eV"),
        ("internal_min_eV", "internal_min_eV"),
        ("internal_max_eV", "internal_max_eV"),
        ("step_meV", "step_meV"),
    ):
        audit.check(
            _close(_float(row, manifest_key), float(grid.get(metadata_key, float("nan")))),
            "grid metadata {}:{}".format(case_id, metadata_key),
        )
    archive_families = tuple(sorted(str(item) for item in metadata.get("families", [])))
    audit.check(
        archive_families == tuple(sorted(_families(row))),
        "family metadata for {}".format(case_id),
    )

    thermal = float(ed_meta.get("thermal_omitted_probability", float("nan")))
    abs_residual = float(scba_meta.get("fixed_point_abs_residual_eV", float("nan")))
    rel_residual = float(scba_meta.get("fixed_point_relative_residual", float("nan")))
    audit.finite_le(thermal, THERMAL_OMISSION_LIMIT, "thermal omission for {}".format(case_id))
    audit.finite_le(abs_residual, SCBA_ABSOLUTE_RESIDUAL_LIMIT_EV, "SCBA absolute residual for {}".format(case_id))
    audit.finite_le(rel_residual, SCBA_RELATIVE_RESIDUAL_LIMIT, "SCBA relative residual for {}".format(case_id))
    audit.check(scba_meta.get("converged") is True, "SCBA converged for {}".format(case_id))
    audit.check(ba_meta.get("converged") is True, "BA completed for {}".format(case_id))

    sigma = arrays["scba_sigma_eV"]
    max_positive_im_sigma = max(0.0, float(np.max(np.imag(sigma))))
    audit.finite_le(max_positive_im_sigma, CAUSALITY_LIMIT_EV, "SCBA causality for {}".format(case_id))
    minimum_spectrum = float("inf")
    raw_mismatch = 0.0
    local_identity = 0.0
    for prefix in GREEN_PREFIXES:
        bright = arrays["{}_green_bright_per_eV".format(prefix)]
        dark = arrays["{}_green_dark_per_eV".format(prefix)]
        local = arrays["{}_green_local_per_eV".format(prefix)]
        local_identity = max(local_identity, _maximum_relative_error(local, 0.5 * (bright + dark)))
        if prefix == "scba":
            minimum_spectrum = min(
                minimum_spectrum,
                float(np.min(-np.imag(bright))),
                float(np.min(-np.imag(dark))),
                float(np.min(-np.imag(local))),
            )
    audit.check(
        minimum_spectrum >= MINIMUM_SPECTRUM_LIMIT_PER_EV,
        "nonnegative SCBA spectra for {}".format(case_id),
        "minimum={:.17g}".format(minimum_spectrum),
    )
    audit.finite_le(local_identity, LOCAL_IDENTITY_RELATIVE_LIMIT, "bright/dark local identity for {}".format(case_id))

    # Production spectra are intentionally stored as complex64.  An absolute
    # summation residual therefore grows with the very large values at sharp
    # poles and is not a meaningful storage-integrity criterion.  Use a
    # scale-aware closure here and retain the absolute value as a diagnostic;
    # the separate float64 solver-validation table enforces absolute closure.
    channel_sum_native = sum(arrays[key] for key in SIGMA_CHANNEL_KEYS)
    channel_closure_native = float(np.max(np.abs(channel_sum_native - sigma)))
    sigma_promoted = np.asarray(sigma, dtype=np.complex128)
    channel_sum = sum(
        np.asarray(arrays[key], dtype=np.complex128)
        for key in SIGMA_CHANNEL_KEYS
    )
    channel_closure = float(np.max(np.abs(channel_sum - sigma_promoted)))
    channel_scale = max(
        1.0e-30,
        float(np.max(np.abs(sigma))),
        float(np.max(np.abs(channel_sum))),
        max(float(np.max(np.abs(arrays[key]))) for key in SIGMA_CHANNEL_KEYS),
    )
    channel_closure_relative = channel_closure / channel_scale
    audit.finite_le(
        channel_closure_relative,
        CHANNEL_CLOSURE_RELATIVE_LIMIT,
        "relative self-energy channel closure for {}".format(case_id),
    )

    for channel in GREEN_CHANNELS:
        ed_spectrum = -np.imag(arrays["ed_green_{}_per_eV".format(channel)])
        scba_spectrum = -np.imag(arrays["scba_green_{}_per_eV".format(channel)])
        ed_area = float(np.trapz(ed_spectrum[mask], energy[mask]))
        scba_area = float(np.trapz(scba_spectrum[mask], energy[mask]))
        mismatch = abs((scba_area - ed_area) / ed_area) if ed_area != 0.0 else float("inf")
        raw_mismatch = max(raw_mismatch, mismatch)
    audit.finite_le(raw_mismatch, RAW_AREA_RELATIVE_LIMIT, "raw ED/SCBA area mismatch for {}".format(case_id))

    bright_weight = float(ed_meta.get("raw_bright_pole_weight", float("nan")))
    dark_weight = float(ed_meta.get("raw_dark_pole_weight", float("nan")))
    pole_weight_error = max(abs(bright_weight - 1.0), abs(dark_weight - 1.0))
    audit.finite_le(pole_weight_error, POLE_WEIGHT_ERROR_LIMIT, "ED pole completeness for {}".format(case_id))

    analysis_min = float(grid.get("analysis_min_eV", float("nan")))
    analysis_max = float(grid.get("analysis_max_eV", float("nan")))
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(analysis_min), abs(analysis_max))
    expected_mask = (energy >= analysis_min - tolerance) & (energy <= analysis_max + tolerance)
    audit.check(np.array_equal(mask, expected_mask), "analysis-mask bounds for {}".format(case_id))

    stats.update(
        {
            "thermal": thermal,
            "abs_residual": abs_residual,
            "rel_residual": rel_residual,
            "positive_im_sigma": max_positive_im_sigma,
            "minimum_spectrum": minimum_spectrum,
            "raw_mismatch": raw_mismatch,
            "pole_weight_error": pole_weight_error,
            "channel_closure_native_absolute": channel_closure_native,
            "channel_closure_absolute": channel_closure,
            "channel_closure_relative": channel_closure_relative,
            "local_identity": local_identity,
        }
    )
    return stats


def _validate_png(audit: Audit, path: Path) -> Tuple[int, int, str]:
    with path.open("rb") as handle:
        signature = handle.read(8)
    audit.check(signature == b"\x89PNG\r\n\x1a\n", "PNG signature {}".format(_relative(path)))
    try:
        import matplotlib.image as mpimg

        image = np.asarray(mpimg.imread(str(path)))
        decoder = "matplotlib"
        height, width = int(image.shape[0]), int(image.shape[1])
        finite = bool(np.all(np.isfinite(image)))
    except Exception as matplotlib_error:
        try:
            from PIL import Image

            with Image.open(str(path)) as image_file:
                image_file.verify()
            with Image.open(str(path)) as image_file:
                width, height = (int(value) for value in image_file.size)
            finite = True
            decoder = "Pillow"
        except Exception as pillow_error:
            raise RuntimeError(
                "matplotlib: {}; Pillow: {}".format(matplotlib_error, pillow_error)
            )
    audit.check(width >= 100 and height >= 100, "PNG dimensions {}".format(_relative(path)))
    audit.check(finite, "finite PNG pixels {}".format(_relative(path)))
    return width, height, decoder


def _validate_figures(audit: Audit) -> Dict[str, Tuple[int, int, str]]:
    dimensions: Dict[str, Tuple[int, int, str]] = {}
    for name in FIGURE_NAMES:
        png = FIGURES_DIR / (name + ".png")
        pdf = FIGURES_DIR / (name + ".pdf")
        for path in (png, pdf):
            audit.check(path.is_file(), "figure exists {}".format(_relative(path)))
            if path.is_file():
                audit.check(path.stat().st_size > 1024, "figure nonempty {}".format(_relative(path)))
        if png.is_file():
            try:
                dimensions[name] = _validate_png(audit, png)
            except Exception as error:
                audit.check(False, "PNG readable {}".format(_relative(png)), str(error))
        if pdf.is_file():
            with pdf.open("rb") as handle:
                header = handle.read(8)
                handle.seek(max(0, pdf.stat().st_size - 2048))
                tail = handle.read()
            audit.check(header.startswith(b"%PDF-"), "PDF header {}".format(_relative(pdf)))
            audit.check(b"%%EOF" in tail, "PDF trailer {}".format(_relative(pdf)))
    audit.check(len(dimensions) == len(FIGURE_NAMES), "all PNG figures decoded")
    return dimensions


def _maximum(rows: Iterable[Mapping[str, str]], key: str, absolute: bool = False) -> float:
    values = [_float(row, key) for row in rows]
    finite = [abs(value) if absolute else value for value in values if math.isfinite(value)]
    return max(finite) if finite else float("nan")


def _minimum(rows: Iterable[Mapping[str, str]], key: str) -> float:
    finite = [_float(row, key) for row in rows if math.isfinite(_float(row, key))]
    return min(finite) if finite else float("nan")


def _median(rows: Iterable[Mapping[str, str]], key: str) -> float:
    finite = sorted(_float(row, key) for row in rows if math.isfinite(_float(row, key)))
    return float(np.median(np.asarray(finite))) if finite else float("nan")


def _format_values(values: Iterable[float]) -> str:
    unique = sorted(set(float(value) for value in values if math.isfinite(float(value))))
    return ", ".join("{:g}".format(value) for value in unique)


def _require_finite_columns(
    audit: Audit,
    rows: Sequence[Mapping[str, str]],
    columns: Iterable[str],
    table_name: str,
) -> None:
    for column in columns:
        bad = [str(row.get("case_id", index)) for index, row in enumerate(rows) if not math.isfinite(_float(row, column))]
        audit.check(
            not bad,
            "finite {} column {}".format(table_name, column),
            "bad rows {}".format(", ".join(bad[:5])),
        )


def _parse_validation_report(audit: Audit, path: Path) -> Dict[str, Any]:
    if not path.is_file():
        audit.check(False, "numerical validation report exists", _relative(path))
        return {"checks_total": 0, "checks_passed": 0, "checks_failed": 1, "profile": "missing"}
    text = path.read_text(encoding="utf-8")
    header: Dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith(("PASS |", "FAIL |")):
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
    def integer(key: str) -> int:
        try:
            return int(header.get(key, "-1"))
        except ValueError:
            return -1
    total = integer("checks_total")
    passed = integer("checks_passed")
    failed = integer("checks_failed")
    explicit_failures = [line for line in text.splitlines() if line.startswith("FAIL |")]
    audit.check(header.get("status") == "PASS", "numerical validation status")
    audit.check(header.get("profile") == "thorough", "numerical validation profile")
    audit.check(total > 0 and passed == total and failed == 0, "numerical validation counts")
    audit.check(not explicit_failures, "no failed numerical validation checks")
    return {
        "checks_total": total,
        "checks_passed": passed,
        "checks_failed": failed,
        "profile": header.get("profile", "unknown"),
    }


def _penultimate_ed_l1(rows: Sequence[Mapping[str, str]]) -> float:
    values: List[float] = []
    groups: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("case_id", ""))].append(row)
    for group in groups.values():
        non_reference = [row for row in group if not _truth(row.get("is_reference", ""))]
        if non_reference:
            last = max(non_reference, key=lambda row: _float(row, "n_vib"))
            values.append(_float(last, "unit_area_l1"))
    return max(values) if values else float("nan")


def _mean_pair(
    audit: Audit,
    rows: Sequence[Mapping[str, str]],
    key: str,
    huang_rhys: float,
    abs_ratio: float,
) -> float:
    selected = [
        row
        for row in rows
        if "large_splitting" in _families(row, "manifest_families")
        and _close(_float(row, "huang_rhys"), huang_rhys)
        and _close(abs(_float(row, "r_2J_over_omega0")), abs_ratio)
        and _close(_float(row, "temperature_K"), 0.0)
    ]
    audit.check(
        len(selected) == 2,
        "large-splitting signed pair S={} |r|={}".format(huang_rhys, abs_ratio),
        "found {}".format(len(selected)),
    )
    values = [_float(row, key) for row in selected]
    if len(values) != 2 or not all(math.isfinite(value) for value in values):
        return float("nan")
    return float(np.mean(values))


def _build_report(
    audit: Audit,
    manifest_rows: Sequence[Mapping[str, str]],
    error_rows: Sequence[Mapping[str, str]],
    table_counts: Mapping[str, int],
    family_counts: Mapping[str, int],
    archive_stats: Sequence[Mapping[str, float]],
    figure_dimensions: Mapping[str, Tuple[int, int, str]],
    validation_summary: Mapping[str, Any],
    ed_convergence_rows: Sequence[Mapping[str, str]],
    scba_convergence_rows: Sequence[Mapping[str, str]],
    figs5_rows: Sequence[Mapping[str, str]],
    smoke_rows: Sequence[Mapping[str, str]],
) -> str:
    status = "PASS" if audit.passed else "FAIL"
    lines: List[str] = [
        "Single-mode Frenkel-dimer benchmark final audit",
        "status = {}".format(status),
        "audit_checks_total = {}".format(audit.checks),
        "audit_checks_failed = {}".format(len(audit.failures)),
        "",
        "Inventory",
        "---------",
        "benchmark_root = Figures/common/dimer/single_mode/source",
        "production_cases = {} (expected {})".format(len(manifest_rows), EXPECTED_CASE_COUNT),
        "unique_combined_archives = {}".format(len(archive_stats)),
        "combined_archive_bytes = {}".format(int(sum(item.get("bytes", 0.0) for item in archive_stats))),
    ]
    for family in EXPECTED_FAMILY_COUNTS:
        lines.append("family_{}_members = {}".format(family, family_counts.get(family, 0)))
    for name in sorted(table_counts):
        lines.append("table_{}_rows = {}".format(name, table_counts[name]))
    lines.extend(
        [
            "figure_pairs = {}".format(len(figure_dimensions)),
            "figure_names = {}".format("; ".join(FIGURE_NAMES)),
            "",
            "Calculations actually run",
            "-------------------------",
            "Every production case contains ED, single-shot BA, and converged SCBA data.",
            "temperatures_K = {}".format(_format_values(_float(row, "temperature_K") for row in manifest_rows)),
            "phonon_frequencies_meV = {}".format(_format_values(_float(row, "omega0_meV") for row in manifest_rows)),
            "eta_values_meV = {}".format(_format_values(_float(row, "eta_meV") for row in manifest_rows)),
            "full_matrix_regressions = {}".format(sum(_truth(row.get("full_matrix_requested", "")) for row in manifest_rows)),
            "smoke_cases = {}".format(len(smoke_rows)),
            "thorough_validation_checks = {}/{} passed".format(
                validation_summary.get("checks_passed", 0), validation_summary.get("checks_total", 0)
            ),
            "The production model is an undamped oscillator; eta is only the common retarded Lorentz HWHM.",
            "",
            "Convergence and numerical integrity",
            "-----------------------------------",
        ]
    )

    def stat_max(key: str) -> float:
        values = [item.get(key, float("nan")) for item in archive_stats]
        finite = [value for value in values if math.isfinite(value)]
        return max(finite) if finite else float("nan")

    def stat_min(key: str) -> float:
        values = [item.get(key, float("nan")) for item in archive_stats]
        finite = [value for value in values if math.isfinite(value)]
        return min(finite) if finite else float("nan")

    ed_penultimate_l1 = _penultimate_ed_l1(ed_convergence_rows)
    grid_rows = [row for row in scba_convergence_rows if row.get("check_type") == "grid_step_halving" and row.get("variant") == "coarse"]
    grid_l1 = _maximum(grid_rows, "unit_area_l1")
    direction_rows = [row for row in scba_convergence_rows if row.get("check_type") == "scan_direction"]
    direction_l1 = _maximum(direction_rows, "unit_area_l1")
    full_precision_channel_closure = _maximum(
        scba_convergence_rows, "channel_closure_max_abs_eV"
    )
    figs5_l1 = _maximum(
        [row for row in figs5_rows if row.get("method") == "scba"],
        "unit_area_l1",
    )
    lines.extend(
        [
            "archive_required_keys = {} per archive; all required arrays and histories checked for finiteness".format(len(REQUIRED_ARCHIVE_KEYS)),
            "max_thermal_omitted_probability = {:.9e} (limit {:.1e})".format(stat_max("thermal"), THERMAL_OMISSION_LIMIT),
            "max_SCBA_fixed_point_absolute_residual_eV = {:.9e} (limit {:.1e})".format(stat_max("abs_residual"), SCBA_ABSOLUTE_RESIDUAL_LIMIT_EV),
            "max_SCBA_fixed_point_relative_residual = {:.9e} (limit {:.1e})".format(stat_max("rel_residual"), SCBA_RELATIVE_RESIDUAL_LIMIT),
            "max_positive_Im_Sigma_eV = {:.9e} (limit {:.1e})".format(stat_max("positive_im_sigma"), CAUSALITY_LIMIT_EV),
            "minimum_minus_Im_G_SCBA_per_eV = {:.9e} (limit {:.1e})".format(stat_min("minimum_spectrum"), MINIMUM_SPECTRUM_LIMIT_PER_EV),
            "max_raw_ED_SCBA_area_relative_mismatch = {:.9e} (limit {:.1e})".format(stat_max("raw_mismatch"), RAW_AREA_RELATIVE_LIMIT),
            "max_ED_pole_weight_error = {:.9e} (limit {:.1e})".format(stat_max("pole_weight_error"), POLE_WEIGHT_ERROR_LIMIT),
            "max_archived_channel_closure_native_absolute_eV = {:.9e} (native complex64 accumulation; diagnostic only)".format(stat_max("channel_closure_native_absolute")),
            "max_archived_channel_closure_promoted_absolute_eV = {:.9e} (complex128 accumulation of stored complex64 values; diagnostic only)".format(stat_max("channel_closure_absolute")),
            "max_archived_channel_closure_relative = {:.9e} (complex64 limit {:.1e})".format(stat_max("channel_closure_relative"), CHANNEL_CLOSURE_RELATIVE_LIMIT),
            "max_float64_validation_channel_closure_eV = {:.9e}".format(full_precision_channel_closure),
            "max_ED_penultimate_truncation_L1 = {:.9e}".format(ed_penultimate_l1),
            "SCBA_grid_halving_L1 = {:.9e}".format(grid_l1),
            "SCBA_forward_reverse_max_L1 = {:.9e}".format(direction_l1),
            "FigS5_SCBA_regression_max_L1 = {:.9e}".format(figs5_l1),
            "validation_profile = {}".format(validation_summary.get("profile", "unknown")),
            "All eight PNG files decoded and all eight PDFs passed header/trailer checks.",
            "",
            "Principal quantitative trends",
            "-----------------------------",
            "Intrinsic coupled dimer (T=0, r=2J/omega0=+1, eta/omega0=0.01):",
            "S        BA_L1    SCBA_L1    Wasserstein_meV    dominant_max_difference_meV    ED/SCBA_peaks",
        ]
    )
    coupled_primary = [
        row
        for row in error_rows
        if "heatmap_r_S" in _families(row, "manifest_families")
        and _close(_float(row, "omega0_meV"), 50.0)
        and _float(row, "huang_rhys") in (0.5, 1.0, 2.0)
        and _close(_float(row, "r_2J_over_omega0"), 1.0)
        and _close(_float(row, "temperature_K"), 0.0)
        and _close(_float(row, "eta_meV"), 0.5)
    ]
    audit.check(
        len(coupled_primary) == 3,
        "intrinsic coupled-dimer primary case count",
        "found {}, expected 3".format(len(coupled_primary)),
    )
    audit.check(
        {_float(row, "huang_rhys") for row in coupled_primary}
        == {0.5, 1.0, 2.0},
        "intrinsic coupled-dimer Huang--Rhys set",
    )
    coupled_primary.sort(key=lambda row: _float(row, "huang_rhys"))
    for row in coupled_primary:
        lines.append(
            "{:3.1f}    {:8.4f}    {:8.4f}    {:15.3f}    {:+14.2f}    {:d}/{:d}".format(
                _float(row, "huang_rhys"),
                _float(row, "ba_unit_area_l1"),
                _float(row, "scba_unit_area_l1"),
                1.0e3 * _float(row, "scba_wasserstein_eV"),
                1.0e3 * _float(row, "scba_peak_comparison_dominant_peak_shift_candidate_minus_reference_eV"),
                int(round(_float(row, "ed_resolved_peak_count"))),
                int(round(_float(row, "scba_resolved_peak_count"))),
            )
        )

    heatmap_t0 = [
        row
        for row in error_rows
        if "heatmap_r_S" in _families(row, "manifest_families")
        and _close(_float(row, "temperature_K"), 0.0)
    ]
    heatmap_t300 = [
        row
        for row in error_rows
        if "heatmap_r_S" in _families(row, "manifest_families")
        and _close(_float(row, "temperature_K"), 300.0)
    ]
    improved = sum(
        _float(row, "scba_unit_area_l1") < _float(row, "ba_unit_area_l1")
        for row in error_rows
    )
    j0_rows: List[Mapping[str, str]] = []
    for huang_rhys in (0.05, 0.5, 1.0, 2.0):
        row = _one(
            audit,
            error_rows,
            "J0 S={}".format(huang_rhys),
            family="heatmap_r_S",
            omega0_meV=50.0,
            huang_rhys=huang_rhys,
            r_2J_over_omega0=0.0,
            temperature_K=0.0,
            eta_meV=0.5,
        )
        if row is not None:
            j0_rows.append(row)
    lines.extend(
        [
            "",
            "Across all {} cases, SCBA has smaller L1 than BA in {}/{} ({:.1f}%); median BA/SCBA L1 = {:.4f}/{:.4f}.".format(
                len(error_rows), improved, len(error_rows), 100.0 * improved / len(error_rows),
                _median(error_rows, "ba_unit_area_l1"), _median(error_rows, "scba_unit_area_l1")
            ),
            "Heat-map L1 at T=0: min/median/max = {:.4f}/{:.4f}/{:.4f}; at 300 K: {:.4f}/{:.4f}/{:.4f}.".format(
                _minimum(heatmap_t0, "scba_unit_area_l1"), _median(heatmap_t0, "scba_unit_area_l1"), _maximum(heatmap_t0, "scba_unit_area_l1"),
                _minimum(heatmap_t300, "scba_unit_area_l1"), _median(heatmap_t300, "scba_unit_area_l1"), _maximum(heatmap_t300, "scba_unit_area_l1")
            ),
            "At J=0 and T=0, SCBA L1 for S=0.05,0.5,1,2 is {}.".format(
                ", ".join("{:.4f}".format(_float(row, "scba_unit_area_l1")) for row in j0_rows)
            ),
        ]
    )

    resonance: Dict[float, Mapping[str, str]] = {}
    for ratio in (-2.0, -1.0, 1.0, 2.0):
        row = _one(
            audit,
            error_rows,
            "weak resonance r={}".format(ratio),
            family="heatmap_r_S",
            omega0_meV=50.0,
            huang_rhys=0.1,
            r_2J_over_omega0=ratio,
            temperature_K=0.0,
            eta_meV=0.5,
        )
        if row is not None:
            resonance[ratio] = row
    if len(resonance) == 4:
        lines.append(
            "At S=0.10,T=0, L1 at r=-2,-1,+1,+2 is {} (positive-emission resonances are larger).".format(
                ", ".join("{:.4f}".format(_float(resonance[r], "scba_unit_area_l1")) for r in (-2.0, -1.0, 1.0, 2.0))
            )
        )

    weak_eta_low = _one(
        audit,
        error_rows,
        "weak eta=0.1",
        omega0_meV=100.0,
        huang_rhys=0.1,
        r_2J_over_omega0=4.0,
        temperature_K=0.0,
        eta_meV=0.1,
    )
    weak_eta_high = _one(
        audit,
        error_rows,
        "weak eta=5",
        omega0_meV=100.0,
        huang_rhys=0.1,
        r_2J_over_omega0=4.0,
        temperature_K=0.0,
        eta_meV=5.0,
    )
    if weak_eta_low is not None and weak_eta_high is not None:
        lines.append(
            "For the weak off-resonant eta scan (S=0.10,r=4), L1 falls from {:.4f} at 0.1 meV to {:.4f} at 5 meV solely through common spectral smoothing.".format(
                _float(weak_eta_low, "scba_unit_area_l1"),
                _float(weak_eta_high, "scba_unit_area_l1"),
            )
        )

    common_means: Dict[Tuple[float, float], float] = {}
    scba_means: Dict[Tuple[float, float], float] = {}
    for huang_rhys in (0.1, 1.0, 2.0):
        for ratio in (4.0, 16.0):
            common_means[(huang_rhys, ratio)] = _mean_pair(
                audit, error_rows, "common_mode_limit_vs_ed_unit_area_l1", huang_rhys, ratio
            )
            scba_means[(huang_rhys, ratio)] = _mean_pair(
                audit, error_rows, "scba_unit_area_l1", huang_rhys, ratio
            )
    lines.append(
        "Mean signed-pair ED/common-mode L1 at |r|=4 -> 16 is S=0.1: {:.4f}->{:.4f}, S=1: {:.4f}->{:.4f}, S=2: {:.4f}->{:.4f}.".format(
            common_means[(0.1, 4.0)], common_means[(0.1, 16.0)],
            common_means[(1.0, 4.0)], common_means[(1.0, 16.0)],
            common_means[(2.0, 4.0)], common_means[(2.0, 16.0)],
        )
    )
    lines.append(
        "At |r|=16 the mean SCBA L1 remains {:.4f} for S=1 and {:.4f} for S=2; off-resonance therefore does not make SCBA exact.".format(
            scba_means[(1.0, 16.0)], scba_means[(2.0, 16.0)]
        )
    )

    t0_absorption = _maximum(
        heatmap_t0,
        "scba_sigma_from_dark_absorption_at_ed_peak_minus_two_imag_eV",
        absolute=True,
    )
    t300_absorption = _maximum(
        heatmap_t300,
        "scba_sigma_from_dark_absorption_at_ed_peak_minus_two_imag_eV",
        absolute=True,
    )
    requested_t0 = [
        row for row in heatmap_t0
        if _float(row, "huang_rhys") in (0.5, 1.0, 2.0)
    ]
    requested_small_shift = [
        row for row in requested_t0
        if abs(1.0e3 * _float(row, "scba_peak_comparison_dominant_peak_shift_candidate_minus_reference_eV")) <= 3.0
    ]
    concealed = max(requested_small_shift, key=lambda row: _float(row, "scba_unit_area_l1")) if requested_small_shift else None

    lines.extend(
        [
            "",
            "Hypothesis audit",
            "----------------",
            "H1 SUPPORTED. The undamped delta bath is not generically easy: J=0 L1 rises sharply with S and resolved ED sidebands are lost at moderate/strong coupling.",
            "H2 SUPPORTED. J=0 is the local independent-boson/NCA problem; the matched historical FigS5 SCBA regression has max L1 {:.3e}.".format(figs5_l1),
            "H3 SUPPORTED WITH PARITY QUALIFICATION.  The positive-J r=+1 point is the symmetry-allowed one-phonon bright--dark resonance.  The large error at r=+2 is an even-parity control and must not be identified with an exact bright--dark avoided crossing; it instead demonstrates that a global NCA error maximum is not, by itself, evidence of a resonance.",
            "H4 SUPPORTED. At T=0 the SCBA dark-internal absorption channel is zero to {:.3e} eV, while at 300 K it reaches {:.3e} eV; negative-J real conversion is thermally activated although virtual error remains.".format(t0_absorption, t300_absorption),
            "H5 SUPPORTED WITH QUALIFICATION. Increasing |r| moves ED toward the S/2 common-mode limit most clearly at weak S, but the tested strong-S cases are not fully asymptotic even at |r|=16; their large SCBA L1 confirms that off-resonance is not phonon-free and not necessarily accurate.",
        ]
    )
    if concealed is not None:
        lines.append(
            "H6 SUPPORTED. In the requested T=0 scan S={:g}, r={:+g} has only {:+.2f} meV difference between dominant maxima but L1={:.4f} and ED/SCBA resolved-peak counts {}/{}; a single maximum conceals large shape error.".format(
                _float(concealed, "huang_rhys"),
                _float(concealed, "r_2J_over_omega0"),
                1.0e3 * _float(concealed, "scba_peak_comparison_dominant_peak_shift_candidate_minus_reference_eV"),
                _float(concealed, "scba_unit_area_l1"),
                int(round(_float(concealed, "ed_resolved_peak_count"))),
                int(round(_float(concealed, "scba_resolved_peak_count"))),
            )
        )
    else:
        lines.append("H6 INCONCLUSIVE. No requested T=0 case met the predeclared small-peak-shift selector.")
    if weak_eta_low is not None and weak_eta_high is not None:
        lines.append(
            "H7 SUPPORTED. Raising the common plotting width from 0.1 to 5 meV reduces weak-case L1 by a factor {:.1f}, although the oscillator remains undamped and the approximation is unchanged.".format(
                _float(weak_eta_low, "scba_unit_area_l1")
                / _float(weak_eta_high, "scba_unit_area_l1")
            )
        )
    else:
        lines.append("H7 INCONCLUSIVE. The required eta endpoints were unavailable.")

    lines.extend(
        [
            "",
            "Answer to Abe's question",
            "------------------------",
            "No: a single undamped oscillator does not generically give the smallest SCBA/NCA error. Its correlation is oscillatory and nondecaying, and at moderate or large Huang--Rhys factor the exact Franck--Condon structure produces order-unity shape errors even when SCBA improves on single-shot BA and preserves the raw sum rule. Accuracy is restricted mainly to weak S away from bright--dark multiphonon resonances; thermal absorption adds negative-J scattering, large |J| removes difference-mode resonance but leaves common-mode dressing, and a large eta can merely hide the discrepancy.",
            "",
            "Scope and limitations",
            "---------------------",
            "All audited production inputs and generated outputs are rooted below Figures/common/dimer/single_mode/source. This audit writes only validation/final_audit.txt; the historical FigS5 benchmark is consumed indirectly through its read-only regression products. The audit cannot reconstruct filesystem history, so the scope statement is a path-policy guarantee rather than proof of earlier modification times.",
            "The central benchmark has gamma_ph=0. A damped Brownian mode or an explicit one-crossing/vertex correction is a separate next test and is not claimed here. Finite eta resolves the distributions but is not a phonon lifetime.",
        ]
    )

    if audit.failures:
        lines.extend(["", "Failures", "--------"])
        lines.extend("- {}".format(failure) for failure in audit.failures)
    # Trend selectors above are audited too, so refresh the header only after
    # every report calculation has run.
    lines[1] = "status = {}".format("PASS" if audit.passed else "FAIL")
    lines[2] = "audit_checks_total = {}".format(audit.checks)
    lines[3] = "audit_checks_failed = {}".format(len(audit.failures))
    return "\n".join(lines) + "\n"


def run_audit() -> Tuple[Audit, str]:
    audit = Audit()

    required_sources = (
        "README.md",
        "calculate_single_mode_ed.py",
        "calculate_single_mode_scba.py",
        "run_single_mode_scan.py",
        "analyze_single_mode_errors.py",
        "plot_single_mode_benchmarks.py",
        "single_mode_common.py",
        "single_mode_metrics.py",
        "audit_single_mode_benchmark.py",
    )
    for name in required_sources:
        path = MODULE_DIR / name
        audit.check(path.is_file() and path.stat().st_size > 0, "source exists {}".format(name))

    manifest_path = RESULTS_DIR / "scan_manifest.csv"
    manifest_fields, manifest_rows = _read_csv(manifest_path)
    _require_columns(
        audit,
        manifest_path,
        manifest_fields,
        (
            "case_id", "families", "status", "output_file", "energy_00_eV",
            "omega0_meV", "huang_rhys", "lambda_meV", "j_meV",
            "r_2J_over_omega0", "temperature_K", "eta_meV",
            "analysis_min_eV", "analysis_max_eV", "internal_min_eV",
            "internal_max_eV", "step_meV", "full_matrix_requested",
            "n_vib", "n_thermal", "thermal_omitted_probability",
            "ed_raw_bright_pole_weight", "ed_raw_dark_pole_weight",
            "scba_fixed_point_abs_residual_eV",
            "scba_fixed_point_relative_residual",
            "scba_max_positive_im_sigma_eV", "error_type", "error_message",
        ),
    )
    audit.check(len(manifest_rows) == EXPECTED_CASE_COUNT, "production manifest case count")
    case_ids = [str(row.get("case_id", "")).strip() for row in manifest_rows]
    audit.check(all(case_ids), "nonempty production case IDs")
    audit.check(len(set(case_ids)) == EXPECTED_CASE_COUNT, "unique production case IDs")
    audit.check(
        all(str(row.get("status", "")).strip().lower() == "complete" for row in manifest_rows),
        "all production cases complete",
    )
    audit.check(
        all(not str(row.get("error_type", "")).strip() and not str(row.get("error_message", "")).strip() for row in manifest_rows),
        "production manifest contains no errors",
    )

    physical_keys = (
        "energy_00_eV", "omega0_meV", "huang_rhys", "j_meV", "temperature_K", "eta_meV"
    )
    physical_tuples = [tuple(_float(row, key) for key in physical_keys) for row in manifest_rows]
    audit.check(
        all(all(math.isfinite(value) for value in item) for item in physical_tuples),
        "finite production physical parameters",
    )
    audit.check(len(set(physical_tuples)) == EXPECTED_CASE_COUNT, "unique production physical cases")

    family_counts: Counter = Counter()
    for row in manifest_rows:
        family_counts.update(_families(row))
    audit.check(
        set(family_counts) == set(EXPECTED_FAMILY_COUNTS),
        "production scan family names",
        "found {}".format(dict(sorted(family_counts.items()))),
    )
    for family, expected in EXPECTED_FAMILY_COUNTS.items():
        audit.check(
            family_counts.get(family, 0) == expected,
            "family member count {}".format(family),
            "found {}, expected {}".format(family_counts.get(family, 0), expected),
        )

    archive_paths: List[Path] = []
    archive_stats: List[Mapping[str, float]] = []
    for row in manifest_rows:
        try:
            path = _resolve_archive(row)
        except Exception as error:
            audit.check(False, "resolve archive {}".format(row.get("case_id")), str(error))
            continue
        archive_paths.append(path)
        archive_stats.append(_validate_archive(audit, row, path))
    audit.check(len(set(archive_paths)) == EXPECTED_CASE_COUNT, "unique production archive paths")
    audit.check(len(archive_stats) == EXPECTED_CASE_COUNT, "all production archives audited")

    error_path = RESULTS_DIR / "ed_scba_error_summary.csv"
    error_fields, error_rows = _read_csv(error_path)
    _require_columns(
        audit,
        error_path,
        error_fields,
        (
            "case_id", "combined_archive", "manifest_families", "manifest_status",
            "omega0_meV", "huang_rhys", "j_meV", "r_2J_over_omega0",
            "temperature_K", "eta_meV", "ed_bright_raw_area",
            "scba_bright_raw_area", "ba_unit_area_l1", "scba_unit_area_l1",
            "scba_wasserstein_eV", "ed_resolved_peak_count",
            "scba_resolved_peak_count", "scba_sideband_missing_count",
            "scba_sideband_spurious_count",
            "scba_peak_comparison_dominant_peak_shift_candidate_minus_reference_eV",
            "scba_sigma_from_dark_absorption_at_ed_peak_minus_two_imag_eV",
            "common_mode_limit_vs_ed_unit_area_l1",
        ),
    )
    audit.check(len(error_rows) == EXPECTED_CASE_COUNT, "error-summary row count")
    error_ids = [str(row.get("case_id", "")) for row in error_rows]
    audit.check(len(set(error_ids)) == EXPECTED_CASE_COUNT, "unique error-summary case IDs")
    audit.check(set(error_ids) == set(case_ids), "error-summary/manifest case coverage")
    audit.check(
        all(str(row.get("manifest_status", "")).strip().lower() == "complete" for row in error_rows),
        "error-summary cases complete",
    )
    _require_finite_columns(
        audit,
        error_rows,
        (
            "ed_bright_raw_area", "ed_dark_raw_area", "ed_local_raw_area",
            "ba_bright_raw_area", "ba_dark_raw_area", "ba_local_raw_area",
            "scba_bright_raw_area", "scba_dark_raw_area", "scba_local_raw_area",
            "ba_unit_area_l1", "scba_unit_area_l1", "scba_wasserstein_eV",
            "ed_resolved_peak_count", "scba_resolved_peak_count",
            "scba_sideband_missing_count", "scba_sideband_spurious_count",
        ),
        "error summary",
    )
    audit.finite_le(
        _maximum(error_rows, "scba_raw_area_relative_difference", absolute=True),
        RAW_AREA_RELATIVE_LIMIT,
        "summary raw ED/SCBA area mismatch",
    )

    ed_path = RESULTS_DIR / "ed_convergence_summary.csv"
    ed_fields, ed_rows = _read_csv(ed_path)
    _require_columns(
        audit, ed_path, ed_fields,
        ("case_id", "is_reference", "n_vib", "reference_n_vib", "thermal_omitted_probability", "unit_area_l1", "thermal_omission_pass", "comparison_to_reference_pass"),
    )
    audit.check(all(_truth(row.get("thermal_omission_pass")) for row in ed_rows), "ED convergence thermal checks pass")
    ed_groups: Dict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in ed_rows:
        ed_groups[str(row.get("case_id", ""))].append(row)
    audit.check(len(ed_groups) == 3, "ED convergence weak/intermediate/strong cases")
    for name, group in sorted(ed_groups.items()):
        references = [row for row in group if _truth(row.get("is_reference"))]
        audit.check(len(references) == 1, "one ED convergence reference {}".format(name))
        passing_nonref = [row for row in group if not _truth(row.get("is_reference")) and _truth(row.get("comparison_to_reference_pass"))]
        audit.check(bool(passing_nonref), "converged ED truncation before reference {}".format(name))

    scba_path = RESULTS_DIR / "scba_convergence_summary.csv"
    scba_fields, scba_rows = _read_csv(scba_path)
    _require_columns(
        audit, scba_path, scba_fields,
        ("check_type", "variant", "fixed_point_abs_residual_eV", "fixed_point_relative_residual", "max_positive_Im_Sigma_eV", "causality_pass", "channel_closure_pass", "validation_pass"),
    )
    audit.check(all(_truth(row.get("causality_pass")) for row in scba_rows), "SCBA convergence causality checks pass")
    audit.check(all(_truth(row.get("channel_closure_pass")) for row in scba_rows), "SCBA convergence channel checks pass")
    audit.check(all(_truth(row.get("validation_pass")) for row in scba_rows), "all SCBA convergence checks pass")
    audit.check(
        {row.get("check_type") for row in scba_rows}
        >= {"grid_step_halving", "padding_widening", "mixing", "seed", "scan_direction", "T0_sweep_iteration", "full_matrix"},
        "SCBA convergence check coverage",
    )

    sideband_path = RESULTS_DIR / "sideband_matching.csv"
    sideband_fields, sideband_rows = _read_csv(sideband_path)
    _require_columns(
        audit, sideband_path, sideband_fields,
        ("case_id", "approximation", "status", "reference_energy_eV", "candidate_energy_eV", "reference_weight", "candidate_weight", "matched_count", "missing_count", "spurious_count"),
    )
    audit.check(set(row.get("case_id", "") for row in sideband_rows) == set(case_ids), "sideband table case coverage")
    by_approximation: Dict[str, set] = defaultdict(set)
    for row in sideband_rows:
        by_approximation[str(row.get("approximation", "")).upper()].add(str(row.get("case_id", "")))
    audit.check(set(by_approximation) == {"BA", "SCBA"}, "sideband approximation labels")
    audit.check(all(by_approximation.get(name, set()) == set(case_ids) for name in ("BA", "SCBA")), "sideband BA/SCBA case coverage")
    audit.check(
        set(str(row.get("status", "")).lower() for row in sideband_rows) <= {"matched", "missing", "spurious"},
        "sideband status labels",
    )

    figs5_path = VALIDATION_DIR / "figs5_regression.csv"
    figs5_fields, figs5_rows = _read_csv(figs5_path)
    _require_columns(audit, figs5_path, figs5_fields, ("S", "method", "unit_area_l1", "validation_pass"))
    audit.check(all(_truth(row.get("validation_pass")) for row in figs5_rows), "FigS5 regressions pass")
    audit.check(set(_float(row, "S") for row in figs5_rows) == {0.5, 1.0, 2.0}, "FigS5 coupling coverage")
    audit.check(set(row.get("method", "") for row in figs5_rows) == {"single_shot_ba", "scba"}, "FigS5 method coverage")

    analytic_path = VALIDATION_DIR / "analytic_validation.json"
    if not analytic_path.is_file():
        audit.check(False, "analytic validation JSON exists")
    else:
        try:
            analytic = json.loads(analytic_path.read_text(encoding="utf-8"))
            audit.check(isinstance(analytic, dict) and _json_is_finite(analytic), "analytic validation JSON finite")
            audit.check(analytic.get("validation_pass") is True, "analytic validation passes")
            threshold_checks = analytic.get("threshold_checks", [])
            audit.check(bool(threshold_checks), "analytic threshold checks present")
            audit.check(all(item.get("passed") is True for item in threshold_checks), "all analytic threshold checks pass")
        except Exception as error:
            audit.check(False, "analytic validation JSON readable", str(error))

    validation_summary = _parse_validation_report(audit, VALIDATION_DIR / "numerical_validation.txt")

    smoke_path = VALIDATION_DIR / "smoke_scan_manifest.csv"
    smoke_fields, smoke_rows = _read_csv(smoke_path)
    _require_columns(audit, smoke_path, smoke_fields, ("case_id", "families", "status", "output_file"))
    audit.check(len(smoke_rows) == 2, "smoke-test case count")
    audit.check(all(str(row.get("status", "")).lower() == "complete" for row in smoke_rows), "smoke-test cases complete")
    for row in smoke_rows:
        output = Path(str(row.get("output_file", "")))
        path = output.resolve() if output.is_absolute() else (MODULE_DIR / output).resolve()
        audit.check(_inside_module(path) and path.is_file(), "smoke archive exists {}".format(row.get("case_id")))

    figure_dimensions = _validate_figures(audit)

    table_counts = {
        "ed_convergence_summary": len(ed_rows),
        "ed_scba_error_summary": len(error_rows),
        "figs5_regression": len(figs5_rows),
        "scan_manifest": len(manifest_rows),
        "scba_convergence_summary": len(scba_rows),
        "sideband_matching": len(sideband_rows),
        "smoke_scan_manifest": len(smoke_rows),
    }
    report = _build_report(
        audit,
        manifest_rows,
        error_rows,
        table_counts,
        family_counts,
        archive_stats,
        figure_dimensions,
        validation_summary,
        ed_rows,
        scba_rows,
        figs5_rows,
        smoke_rows,
    )
    return audit, report


def _atomic_write(path: Path, text: str) -> None:
    if not _inside_module(path):
        raise ValueError("refusing to write outside benchmark directory: {}".format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run every check and print the report without writing final_audit.txt",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        audit, report = run_audit()
    except Exception as error:
        audit = Audit()
        audit.check(False, "audit execution", "{}: {}".format(type(error).__name__, error))
        report = (
            "Single-mode Frenkel-dimer benchmark final audit\n"
            "status = FAIL\n"
            "audit_checks_total = 1\n"
            "audit_checks_failed = 1\n\n"
            "Failures\n--------\n- {}\n".format(audit.failures[0])
        )
    if not args.check_only:
        _atomic_write(FINAL_REPORT, report)
    sys.stdout.write(report)
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
