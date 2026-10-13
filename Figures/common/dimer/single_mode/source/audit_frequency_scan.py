#!/usr/bin/env python3
"""Independent final audit for the controlled single-mode frequency scan.

The audit is deliberately read-only except for its deterministic text report,
``validation/frequency_scan/final_audit.txt``.  Every path is resolved from
this file, so the command is independent of the current working directory::

    python3 audit_frequency_scan.py --check-only
    python3 audit_frequency_scan.py

``--check-only`` performs exactly the same checks but does not write the
report.  A failed check always gives a nonzero exit status.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
RESULT_DIR = MODULE_DIR / "results" / "frequency_scan"
VALIDATION_DIR = MODULE_DIR / "validation" / "frequency_scan"
FIGURE_DIR = MODULE_DIR / "figures" / "frequency_scan"
REPORT_PATH = VALIDATION_DIR / "final_audit.txt"

SCHEMA_VERSION = "frequency_scan_v1"
PRIMARY_S = (0.02, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0)
EXPECTED_PRIMARY_R_POINTS = 369
EXPECTED_SECONDARY_R_POINTS = 26
LEGACY_PRIMARY_R_POINTS = 167
EXPECTED_FAMILIES = {
    "primary_fixed_S": len(PRIMARY_S) * 2 * EXPECTED_PRIMARY_R_POINTS,
    "primary_J0_baseline": len(PRIMARY_S),
    "secondary_fixed_lambda": 2 * EXPECTED_SECONDARY_R_POINTS,
    "secondary_J0_baseline": EXPECTED_SECONDARY_R_POINTS,
}
EXPECTED_TOTAL = sum(EXPECTED_FAMILIES.values())
SIGNIFICANT_WEIGHT = 0.01

SELECTED_LINE_SCAN_S = (0.02, 0.05, 0.10)
INTEGER_GUIDES = (1, 2, 3, 4, 5)
ACTIVE_FIGURE_STEMS = {
    "wasserstein_line_scan",
    "resonance_sweep_spectra",
}
COMMAND15_FIGURE_STEMS = {
    "exact_level_strength_map",
    "representative_spectra",
}
COMMAND14_FIGURE_STEMS = {
    "frequency_resonance_map",
    "selected_frequency_spectra",
}
EXPECTED_ARCHIVE_COUNT = 26
REPRESENTATIVE_COLUMNS = (
    "display_order", "panel", "case_id", "role", "huang_rhys",
    "j_sign", "j_meV", "r_abs", "omega0_meV", "eta_over_omega0",
    "scba_w_over_omega0", "ed_raw_area", "scba_raw_area",
    "scba_raw_area_relative_difference", "archive_file",
)
RESONANCE_SWEEP_COLUMNS = (
    "display_order", "figure_panel", "cut_label", "role", "selection_rule",
    "case_id", "huang_rhys", "j_sign", "j_meV", "r_abs", "omega0_meV",
    "eta_over_omega0", "scba_w_over_omega0",
    "local_baseline_w_over_omega0", "ed_raw_area", "scba_raw_area",
    "scba_raw_area_relative_difference", "archive_file",
)
EXPECTED_RESONANCE_SWEEP = (
    (0.05, 0.65, "c", "A", "lower flank"),
    (0.05, 0.88, "c", "B", "measured maximum"),
    (0.05, 1.00, "c", "C", "bare guide"),
    (0.05, 1.45, "c", "D", "upper off-integer reference minimum"),
    (0.10, 0.65, "d", "A", "lower flank"),
    (0.10, 0.80, "d", "B", "measured maximum"),
    (0.10, 1.00, "d", "C", "bare guide"),
    (0.10, 1.35, "d", "D", "upper off-integer reference minimum"),
)
LINE_SCAN_SOURCE_COLUMNS = (
    "case_id", "huang_rhys", "j_sign", "j_meV", "r_abs", "omega0_meV",
    "eta_over_omega0", "scba_w_over_omega0",
)
PEAK_MANIFEST_COLUMNS = (
    "huang_rhys", "j_sign", "integer_guide", "feature_label",
    "extremum_type", "r_peak", "w_peak", "peak_case_id",
    "r_reference_min", "w_reference_min", "reference_case_id",
    "reference_label", "search_window", "reference_window",
)
SOURCE_MANIFEST_COLUMNS = (
    "figure_stem", "panels", "output_png", "output_pdf", "source_file",
    "source_sha256", "source_size_bytes", "selection", "plotted_quantities",
    "transformation", "notes",
)
LEGACY_FREQUENCY_MAP_COLUMNS = (
    "case_id", "huang_rhys", "j_sign", "j_meV", "r_abs", "omega0_meV",
    "eta_over_omega0", "scba_w_over_omega0", "scba_sideband_weight_error",
    "scba_significant_missing_count", "scba_significant_spurious_count",
    "valid_wasserstein", "valid_sideband_weight", "valid_peak_topology",
    "operational_validity_pass",
)
LEGACY_REPRESENTATIVE_COLUMNS = (
    "display_order", "panel", "case_id", "role", "huang_rhys", "j_sign",
    "j_meV", "r_abs", "omega0_meV", "eta_over_omega0", "scba_l1",
    "scba_w_over_omega0", "tracked_scba_shift_over_omega0",
    "tracked_scba_weight_error", "tracked_ed_ambiguity",
    "tracked_scba_ambiguity", "scba_sideband_weight_error",
    "scba_missing_count", "scba_spurious_count",
    "scba_significant_missing_count", "scba_significant_spurious_count",
    "ed_raw_area", "ba_raw_area", "scba_raw_area",
    "ba_raw_area_relative_difference", "scba_raw_area_relative_difference",
    "operational_validity_pass", "archive_file",
)

# Numerical-integrity limits.  The accuracy gate is loaded from the generated
# validation JSON and independently checked against its declared values.
RAW_AREA_LIMIT = 3.0e-6
ED_POLE_WEIGHT_LIMIT = 3.0e-12
ED_MOMENT_LIMIT = 3.0e-10
SCBA_ABS_RESIDUAL_OVER_OMEGA_LIMIT = 2.0e-12
SCBA_REL_RESIDUAL_LIMIT = 1.0e-8
SCBA_CAUSALITY_OVER_OMEGA_LIMIT = 2.0e-12
SPECTRUM_NEGATIVE_LIMIT = -1.0e-8
CHANNEL_CLOSURE_OVER_OMEGA_LIMIT = 1.0e-10
ARCHIVE_CHANNEL_CLOSURE_RELATIVE_LIMIT = 2.0e-7

EXPECTED_ARCHIVE_KEYS = (
    "energy_eV",
    "analysis_mask",
    "ed_green_bright_per_eV",
    "ba_green_bright_per_eV",
    "scba_green_bright_per_eV",
    "scba_sigma_eV",
    "scba_sigma_from_bright_emission_eV",
    "scba_sigma_from_bright_absorption_eV",
    "scba_sigma_from_dark_emission_eV",
    "scba_sigma_from_dark_absorption_eV",
    "metadata_json",
)


class Audit:
    """Accumulate deterministic checks and compact failure details."""

    def __init__(self) -> None:
        self.checks = 0
        self.failures: List[str] = []
        self.notes: List[str] = []

    def check(self, condition: bool, name: str, detail: str = "") -> bool:
        self.checks += 1
        passed = bool(condition)
        if not passed:
            message = name
            if detail:
                message += ": " + detail
            self.failures.append(message)
        return passed

    def note(self, text: str) -> None:
        self.notes.append(text)


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "pass", "passed")


def _float(row: Mapping[str, Any], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _int(row: Mapping[str, Any], key: str) -> int:
    value = _float(row, key)
    if not np.isfinite(value):
        return -1
    return int(round(value))


def _close(left: float, right: float, tolerance: float = 1.0e-10) -> bool:
    return bool(
        np.isfinite(left)
        and np.isfinite(right)
        and abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))
    )


def _load_csv(
    audit: Audit,
    path: Path,
    required_columns: Sequence[str] = (),
    allow_empty: bool = False,
) -> List[Dict[str, str]]:
    if not audit.check(path.is_file(), "file exists", str(path)):
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            rows = [dict(row) for row in reader]
    except (OSError, csv.Error, UnicodeError) as error:
        audit.check(False, "CSV readable", "{}: {}".format(path, error))
        return []
    audit.check(bool(fields), "CSV has header", str(path))
    missing = [column for column in required_columns if column not in fields]
    audit.check(not missing, "CSV required columns", "{} missing {}".format(path, missing))
    audit.check(allow_empty or bool(rows), "CSV nonempty", str(path))
    return rows


def _audit_exact_csv_schema(
    audit: Audit,
    path: Path,
    expected_columns: Sequence[str],
    label: str,
) -> None:
    """Require an exact, ordered header for compact derived products."""

    if not path.is_file():
        return
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            fields = tuple(csv.DictReader(handle).fieldnames or ())
    except (OSError, csv.Error, UnicodeError) as error:
        audit.check(False, label + " schema readable", "{}: {}".format(path, error))
        return
    audit.check(
        fields == tuple(expected_columns),
        label + " exact column schema",
        "observed {}, expected {}".format(fields, tuple(expected_columns)),
    )


def _finite_columns(
    audit: Audit,
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    label: str,
) -> None:
    bad: List[str] = []
    for row_index, row in enumerate(rows, start=2):
        for column in columns:
            if not np.isfinite(_float(row, column)):
                bad.append("row {} {}={!r}".format(row_index, column, row.get(column)))
                if len(bad) >= 12:
                    break
        if len(bad) >= 12:
            break
    audit.check(not bad, label + " finite numeric fields", "; ".join(bad))


def _read_json(audit: Audit, path: Path) -> Dict[str, Any]:
    if not audit.check(path.is_file(), "file exists", str(path)):
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        audit.check(False, "JSON readable", "{}: {}".format(path, error))
        return {}
    audit.check(isinstance(value, dict), "JSON top level is object", str(path))
    return value if isinstance(value, dict) else {}


def _primary_r_grid() -> Tuple[float, ...]:
    """Command-14 mesh unioned with 0.005 windows at all five integers."""

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


def _legacy_primary_r_grid() -> Tuple[float, ...]:
    """Frozen command-14 mesh retained by the union and legacy CSV."""

    hundredths = set(range(25, 551, 10))
    hundredths.add(550)
    for center in (100, 300, 500):
        hundredths.update(range(center - 20, center + 21))
    hundredths.update((100, 200, 300, 400, 500))
    return tuple(value / 100.0 for value in sorted(hundredths) if 25 <= value <= 550)


def _audit_manifest_and_summary(
    audit: Audit,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    manifest_columns = (
        "schema_version", "case_id", "family", "status", "definition_key",
        "archive_file", "save_archive", "energy_00_eV", "omega0_meV",
        "huang_rhys", "lambda_meV", "j_meV", "j_sign", "r_abs",
        "r_signed", "temperature_K", "eta_meV", "eta_over_omega0",
        "analysis_x_min", "analysis_x_max", "internal_x_min",
        "internal_x_max", "dx", "grid_samples", "n_vib",
    )
    summary_columns = (
        "schema_version", "case_id", "family", "omega0_meV", "huang_rhys",
        "lambda_meV", "j_meV", "j_sign", "r_abs", "r_signed",
        "temperature_K", "eta_meV", "eta_over_omega0", "dx", "n_vib",
        "ed_raw_area", "scba_raw_area", "scba_raw_area_relative_difference",
        "scba_l1", "scba_w_over_omega0", "scba_sideband_weight_error",
        "scba_significant_missing_count", "scba_significant_spurious_count",
        "significant_basin_weight_threshold", "tracked_scba_shift_over_omega0",
        "tracked_scba_weight_error", "tracked_ed_ambiguity",
        "tracked_scba_ambiguity", "ed_pole_weight_error",
        "ed_common_omitted_weight", "ed_first_moment_error_over_omega0",
        "ed_minimum_spectrum", "scba_fixed_point_abs_residual_over_omega0",
        "scba_fixed_point_relative_residual", "scba_max_positive_im_sigma_over_omega0",
        "scba_minimum_spectrum", "scba_channel_closure_over_omega0",
        "valid_l1", "valid_wasserstein", "valid_tracked_shift",
        "valid_tracked_weight", "valid_sideband_weight", "valid_peak_topology",
        "valid_raw_area", "valid_tracking", "operational_validity_pass",
    )
    manifest = _load_csv(audit, RESULT_DIR / "manifest.csv", manifest_columns)
    summary = _load_csv(audit, RESULT_DIR / "summary.csv", summary_columns)

    audit.check(len(manifest) == EXPECTED_TOTAL, "manifest row count",
                "observed {}, expected {}".format(len(manifest), EXPECTED_TOTAL))
    audit.check(len(summary) == EXPECTED_TOTAL, "summary row count",
                "observed {}, expected {}".format(len(summary), EXPECTED_TOTAL))

    manifest_ids = [row.get("case_id", "") for row in manifest]
    summary_ids = [row.get("case_id", "") for row in summary]
    audit.check(all(manifest_ids), "manifest case IDs nonempty")
    audit.check(len(set(manifest_ids)) == len(manifest_ids), "manifest case IDs unique")
    audit.check(len(set(summary_ids)) == len(summary_ids), "summary case IDs unique")
    audit.check(set(manifest_ids) == set(summary_ids), "manifest/summary ID coverage",
                "missing summary={}, orphan summary={}".format(
                    len(set(manifest_ids) - set(summary_ids)),
                    len(set(summary_ids) - set(manifest_ids)),
                ))

    family_counts = Counter(row.get("family", "") for row in manifest)
    audit.check(dict(family_counts) == EXPECTED_FAMILIES, "manifest family inventory",
                "observed {}".format(dict(sorted(family_counts.items()))))
    summary_family_counts = Counter(row.get("family", "") for row in summary)
    audit.check(dict(summary_family_counts) == EXPECTED_FAMILIES, "summary family inventory",
                "observed {}".format(dict(sorted(summary_family_counts.items()))))
    audit.check(all(row.get("status") == "complete" for row in manifest),
                "all manifest cases complete",
                "statuses {}".format(dict(Counter(row.get("status", "") for row in manifest))))
    audit.check(all(row.get("started_utc") and row.get("finished_utc") for row in manifest),
                "all complete manifest cases have timestamps")
    audit.check(all(not row.get("error_type") and not row.get("error_message") for row in manifest),
                "manifest contains no recorded errors")
    audit.check(all(row.get("schema_version") == SCHEMA_VERSION for row in manifest),
                "manifest schema version")
    audit.check(all(row.get("schema_version") == SCHEMA_VERSION for row in summary),
                "summary schema version")

    definition_keys = [row.get("definition_key", "") for row in manifest]
    audit.check(all(definition_keys), "definition keys nonempty")
    audit.check(len(set(definition_keys)) == len(definition_keys), "definition keys unique")
    invalid_definition: List[str] = []
    for row in manifest:
        try:
            definition = json.loads(row.get("definition_key", ""))
            valid = definition.get("schema") == SCHEMA_VERSION
        except (json.JSONDecodeError, AttributeError):
            valid = False
        if not valid:
            invalid_definition.append(row.get("case_id", ""))
    audit.check(not invalid_definition, "definition keys parse and match schema",
                ", ".join(invalid_definition[:10]))

    manifest_numeric = (
        "energy_00_eV", "omega0_meV", "huang_rhys", "lambda_meV", "j_meV",
        "j_sign", "r_abs", "r_signed", "temperature_K", "eta_meV",
        "eta_over_omega0", "analysis_x_min", "analysis_x_max",
        "internal_x_min", "internal_x_max", "dx", "grid_samples", "n_vib",
        "runtime_s",
    )
    _finite_columns(audit, manifest, manifest_numeric, "manifest")

    expected_r = set(_primary_r_grid())
    audit.check(len(expected_r) == EXPECTED_PRIMARY_R_POINTS, "internal primary r-grid definition")
    primary = [row for row in manifest if row.get("family") == "primary_fixed_S"]
    primary_groups: DefaultDict[Tuple[float, int], Set[float]] = defaultdict(set)
    parameter_failures: List[str] = []
    for row in primary:
        s_value = _float(row, "huang_rhys")
        sign = _int(row, "j_sign")
        r_abs = _float(row, "r_abs")
        omega = _float(row, "omega0_meV")
        primary_groups[(s_value, sign)].add(r_abs)
        valid = (
            s_value in PRIMARY_S
            and sign in (-1, 1)
            and _close(abs(_float(row, "j_meV")), 25.0)
            and _close(omega, 50.0 / r_abs)
            and _close(_float(row, "r_signed"), sign * r_abs)
            and _close(_float(row, "lambda_meV"), s_value * omega)
            and _close(_float(row, "eta_meV"), 0.01 * omega)
            and _close(_float(row, "eta_over_omega0"), 0.01)
            and _close(_float(row, "temperature_K"), 0.0)
        )
        if not valid:
            parameter_failures.append(row.get("case_id", ""))
    audit.check(not parameter_failures, "primary parameter identities",
                ", ".join(parameter_failures[:10]))
    audit.check(set(primary_groups) == {(s_value, sign) for s_value in PRIMARY_S for sign in (-1, 1)},
                "primary S/sign group coverage")
    bad_grids = [
        "S={} sign={} n={} missing={} extra={}".format(
            key[0], key[1], len(values), len(expected_r - values), len(values - expected_r)
        )
        for key, values in sorted(primary_groups.items()) if values != expected_r
    ]
    audit.check(not bad_grids, "primary r-grid coverage", "; ".join(bad_grids[:10]))

    primary_baselines = [row for row in manifest if row.get("family") == "primary_J0_baseline"]
    audit.check({ _float(row, "huang_rhys") for row in primary_baselines } == set(PRIMARY_S),
                "primary baseline S coverage")
    audit.check(all(
        _close(_float(row, "j_meV"), 0.0)
        and _close(_float(row, "omega0_meV"), 50.0)
        and _close(_float(row, "eta_meV"), 0.5)
        and _close(_float(row, "r_signed"), 0.0)
        for row in primary_baselines
    ), "primary baseline parameters")

    secondary = [row for row in manifest if row.get("family") == "secondary_fixed_lambda"]
    secondary_r = { _float(row, "r_abs") for row in secondary }
    audit.check(len(secondary_r) == EXPECTED_SECONDARY_R_POINTS,
                "fixed-lambda r-grid size", "observed {}".format(len(secondary_r)))
    secondary_groups = Counter((_float(row, "r_abs"), _int(row, "j_sign")) for row in secondary)
    audit.check(all(secondary_groups[(r_abs, sign)] == 1 for r_abs in secondary_r for sign in (-1, 1)),
                "fixed-lambda sign/r coverage")
    audit.check(all(
        _close(_float(row, "lambda_meV"), 30.0)
        and _close(abs(_float(row, "j_meV")), 25.0)
        and _close(_float(row, "omega0_meV"), 50.0 / _float(row, "r_abs"))
        and _close(_float(row, "huang_rhys"), 0.6 * _float(row, "r_abs"))
        and _close(_float(row, "eta_over_omega0"), 0.01)
        for row in secondary
    ), "fixed-lambda parameter identities")

    secondary_baselines = [row for row in manifest if row.get("family") == "secondary_J0_baseline"]
    audit.check({ _float(row, "r_abs") for row in secondary_baselines } == secondary_r,
                "fixed-lambda baseline r coverage")
    audit.check(all(
        _close(_float(row, "j_meV"), 0.0)
        and _close(_float(row, "lambda_meV"), 30.0)
        and _close(_float(row, "eta_over_omega0"), 0.01)
        for row in secondary_baselines
    ), "fixed-lambda baseline parameters")

    summary_by_id = {row.get("case_id", ""): row for row in summary}
    manifest_by_id = {row.get("case_id", ""): row for row in manifest}
    mismatch: List[str] = []
    for case_id in sorted(set(manifest_by_id) & set(summary_by_id)):
        left = manifest_by_id[case_id]
        right = summary_by_id[case_id]
        if left.get("family") != right.get("family"):
            mismatch.append(case_id + ":family")
            continue
        for column in ("omega0_meV", "huang_rhys", "lambda_meV", "j_meV", "j_sign",
                       "r_abs", "r_signed", "temperature_K", "eta_meV", "eta_over_omega0",
                       "dx", "n_vib"):
            if not _close(_float(left, column), _float(right, column), 2.0e-10):
                mismatch.append(case_id + ":" + column)
                break
    audit.check(not mismatch, "manifest/summary metadata agreement", ", ".join(mismatch[:12]))

    # This is the compact, figure-facing baseline table.  It is expected to be
    # an exact projection of the 7 fixed-S and 26 fixed-lambda J=0 rows, not a
    # separately recomputed data product.
    baseline_rows = _load_csv(
        audit,
        RESULT_DIR / "local_baselines.csv",
        ("schema_version", "case_id", "family", "j_meV", "scba_l1",
         "scba_w_over_omega0", "operational_validity_pass"),
    )
    expected_baseline_ids = {
        case_id for case_id, row in summary_by_id.items()
        if row.get("family") in ("primary_J0_baseline", "secondary_J0_baseline")
    }
    baseline_ids = [row.get("case_id", "") for row in baseline_rows]
    audit.check(len(baseline_rows) == 33, "local-baseline row count",
                "observed {}, expected 33".format(len(baseline_rows)))
    audit.check(len(set(baseline_ids)) == len(baseline_ids), "local-baseline IDs unique")
    audit.check(set(baseline_ids) == expected_baseline_ids,
                "local-baseline case coverage")
    baseline_mismatch = [
        case_id for case_id in baseline_ids
        if case_id not in summary_by_id or baseline_rows[baseline_ids.index(case_id)] != summary_by_id[case_id]
    ]
    audit.check(not baseline_mismatch, "local baselines exactly trace to summary",
                ", ".join(baseline_mismatch[:12]))
    return manifest, summary, summary_by_id


def _audit_cutoffs(audit: Audit, manifest: Sequence[Mapping[str, str]]) -> Dict[str, int]:
    columns = (
        "family", "case_id", "huang_rhys", "n_low", "n_high",
        "stick_wasserstein_over_omega0", "maximum_weight_missing",
        "first_moment_error_over_omega0", "pass",
    )
    rows = _load_csv(audit, VALIDATION_DIR / "ed_cutoff_convergence.csv", columns)
    _finite_columns(audit, rows, (
        "huang_rhys", "n_low", "n_high", "stick_wasserstein_over_omega0",
        "maximum_weight_missing", "first_moment_error_over_omega0",
    ), "ED cutoff convergence")
    audit.check(all(_int(row, "n_high") == _int(row, "n_low") + 2 for row in rows),
                "ED cutoff increments are two")
    audit.check(all(_int(row, "n_high") <= 44 for row in rows), "ED cutoff cap")
    cutoff_keys = [
        (row.get("family", ""), row.get("case_id", ""), _float(row, "huang_rhys"),
         _int(row, "n_low"), _int(row, "n_high"))
        for row in rows
    ]
    audit.check(len(set(cutoff_keys)) == len(cutoff_keys), "ED cutoff rows unique")

    primary_rows = [row for row in rows if row.get("family") == "primary_fixed_S"]
    primary_groups: DefaultDict[float, List[Mapping[str, str]]] = defaultdict(list)
    for row in primary_rows:
        primary_groups[_float(row, "huang_rhys")].append(row)
    audit.check(set(primary_groups) == set(PRIMARY_S), "primary cutoff S coverage")
    audit.check(all(sum(_truth(row.get("pass")) for row in group) == 1
                    and _truth(max(group, key=lambda item: _int(item, "n_high")).get("pass"))
                    for group in primary_groups.values()),
                "every primary S ends at one passing adaptive cutoff")
    primary_selected = {
        s_value: _int(next(row for row in group if _truth(row.get("pass"))), "n_high")
        for s_value, group in primary_groups.items()
        if sum(_truth(row.get("pass")) for row in group) == 1
    }
    audit.check(all(
        _int(row, "n_vib") == primary_selected.get(_float(row, "huang_rhys"), -1)
        for row in manifest
        if row.get("family") in ("primary_fixed_S", "primary_J0_baseline")
    ), "primary production rows use selected adaptive cutoffs")

    expected_secondary_ids = {
        row.get("case_id", "") for row in manifest
        if row.get("family") in ("secondary_fixed_lambda", "secondary_J0_baseline")
    }
    secondary_groups: DefaultDict[str, List[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("family") in ("secondary_fixed_lambda", "secondary_J0_baseline"):
            secondary_groups[row.get("case_id", "")].append(row)
    audit.check(set(secondary_groups) == expected_secondary_ids,
                "secondary cutoff case coverage",
                "observed {}, expected {}".format(len(secondary_groups), len(expected_secondary_ids)))
    audit.check(all(sum(_truth(row.get("pass")) for row in group) == 1
                    and _truth(max(group, key=lambda item: _int(item, "n_high")).get("pass"))
                    for group in secondary_groups.values()),
                "every secondary case ends at one passing adaptive cutoff")
    secondary_selected = {
        case_id: _int(next(row for row in group if _truth(row.get("pass"))), "n_high")
        for case_id, group in secondary_groups.items()
        if sum(_truth(row.get("pass")) for row in group) == 1
    }
    audit.check(all(
        _int(row, "n_vib") == secondary_selected.get(row.get("case_id", ""), -1)
        for row in manifest
        if row.get("family") in ("secondary_fixed_lambda", "secondary_J0_baseline")
    ), "secondary production rows use selected adaptive cutoffs")

    passing = [row for row in rows if _truth(row.get("pass"))]
    audit.check(all(
        _float(row, "stick_wasserstein_over_omega0") <= 2.0e-9
        and _float(row, "maximum_weight_missing") <= 2.0e-12
        and _float(row, "first_moment_error_over_omega0") <= 2.0e-10
        for row in passing
    ), "passing cutoff rows satisfy declared thresholds")
    return {"rows": len(rows), "passing_rows": len(passing)}


def _audit_exact_parity(audit: Audit) -> Dict[str, Any]:
    level_columns = (
        "schema_version", "huang_rhys", "r_abs", "omega0_meV", "n_vib",
        "level_index", "parity", "energy_minus_epsilon_v_over_omega0",
        "detuning_from_bare_bright_over_omega0", "difference_mode_energy_eV",
        "bright0_weight", "bright_electronic_population", "dark_electronic_population",
        "mean_difference_phonon", "dominant_basis_weight", "cutoff_edge_population",
        "eigen_residual",
    )
    levels = _load_csv(audit, RESULT_DIR / "exact_level_map.csv", level_columns)
    audit.check(len(levels) == 8688, "exact level-map row count",
                "observed {}, expected 8688".format(len(levels)))
    audit.check(all(row.get("schema_version") == "1" for row in levels),
                "exact level-map schema version")
    _finite_columns(audit, levels, level_columns[1:], "exact level map")
    audit.check({ _float(row, "huang_rhys") for row in levels } == {0.10, 0.50, 1.0},
                "exact level-map S coverage")
    level_groups = Counter((_float(row, "huang_rhys"), _float(row, "r_abs")) for row in levels)
    audit.check(len(level_groups) == 3 * 181, "exact level-map S/r coverage",
                "observed {} groups".format(len(level_groups)))
    audit.check(all(count == 16 for count in level_groups.values()),
                "exact level-map saves 16 levels per S/r")
    audit.check(all(_int(row, "parity") in (-1, 1) for row in levels),
                "exact level-map parity labels")
    audit.check(max(abs(_float(row, "eigen_residual")) for row in levels) <= 5.0e-12,
                "exact level-map eigen residual")
    audit.check(all(
        -1.0e-12 <= _float(row, "bright0_weight") <= 1.0 + 1.0e-12
        and abs(_float(row, "bright_electronic_population")
                + _float(row, "dark_electronic_population") - 1.0) <= 2.0e-10
        for row in levels
    ), "exact level-map probabilities")

    crossing_columns = (
        "schema_version", "crossing_type", "huang_rhys", "order_m",
        "parity_B0", "parity_Dm", "same_parity", "tracked_gap_ambiguous",
        "direct_matrix_element_over_omega0", "forbidden_block_coupling_over_omega0",
    )
    crossings = _load_csv(audit, RESULT_DIR / "exact_level_map_crossings.csv", crossing_columns)
    audit.check(len(crossings) == 15, "exact crossing row count",
                "observed {}, expected 15".format(len(crossings)))
    audit.check(all(row.get("schema_version") == "1" for row in crossings),
                "exact crossing schema version")
    crossing_types = Counter(row.get("crossing_type", "") for row in crossings)
    audit.check(crossing_types == {"odd_allowed": 9, "even_forbidden_control": 6},
                "exact crossing inventory", str(dict(crossing_types)))
    parity_failure: List[str] = []
    max_forbidden = 0.0
    for row in crossings:
        order = _int(row, "order_m")
        same = _truth(row.get("same_parity"))
        if order % 2 == 1:
            valid = same and _int(row, "parity_B0") == _int(row, "parity_Dm") == 1
        else:
            coupling = _float(row, "forbidden_block_coupling_over_omega0")
            max_forbidden = max(max_forbidden, abs(coupling)) if np.isfinite(coupling) else float("inf")
            valid = (not same and _int(row, "parity_B0") == 1
                     and _int(row, "parity_Dm") == -1 and np.isfinite(coupling))
        if not valid:
            parity_failure.append("S={} m={}".format(row.get("huang_rhys"), order))
    audit.check(not parity_failure, "odd/even Rabi-parity convention", ", ".join(parity_failure))
    audit.check(max_forbidden <= 2.0e-13, "forbidden parity-block coupling",
                "max {:.9e}".format(max_forbidden))

    validation = _load_csv(
        audit,
        VALIDATION_DIR / "exact_level_map_validation.csv",
        ("schema_version", "test_name", "observed", "expected", "absolute_error",
         "tolerance", "passed", "ambiguous"),
    )
    audit.check(len(validation) == 99, "exact parity validation row count",
                "observed {}, expected 99".format(len(validation)))
    audit.check(all(row.get("schema_version") == "1" for row in validation),
                "exact parity validation schema version")
    audit.check(all(_truth(row.get("passed")) for row in validation),
                "all 99 exact parity validations pass")
    # Predicate tests deliberately have no scalar target: their expected value
    # and absolute error are NaN by schema.  A missing targeted even crossing is
    # likewise an explicitly ambiguous, predicate-only result with no tolerance.
    predicate_names = {"odd_parity_allowed_gap", "even_target_crossing_not_in_scan"}
    validation_numeric_bad: List[str] = []
    for index, row in enumerate(validation, start=2):
        observed = _float(row, "observed")
        expected = _float(row, "expected")
        error = _float(row, "absolute_error")
        tolerance = _float(row, "tolerance")
        name = row.get("test_name", "")
        if not np.isfinite(observed):
            validation_numeric_bad.append("row {} nonfinite observed".format(index))
        if np.isfinite(expected):
            if not (np.isfinite(error) and np.isfinite(tolerance)):
                validation_numeric_bad.append("row {} incomplete scalar comparison".format(index))
        elif not (math.isnan(expected) and math.isnan(error) and name in predicate_names):
            validation_numeric_bad.append("row {} invalid predicate NaNs".format(index))
        elif name == "odd_parity_allowed_gap" and not np.isfinite(tolerance):
            validation_numeric_bad.append("row {} odd-gap tolerance nonfinite".format(index))
        elif name == "even_target_crossing_not_in_scan" and not (
            math.isnan(tolerance) and _truth(row.get("ambiguous"))
        ):
            validation_numeric_bad.append("row {} invalid out-of-scan marker".format(index))
    audit.check(not validation_numeric_bad, "exact parity validation numeric schema",
                "; ".join(validation_numeric_bad[:12]))

    _audit_npz_finite(audit, RESULT_DIR / "exact_level_map.npz", required=(
        "huang_rhys", "r_abs", "energy_minus_epsilon_v_over_omega0",
        "bright0_weight", "parity", "metadata_json",
    ))
    _audit_npz_finite(audit, VALIDATION_DIR / "exact_level_map_validation.npz", required=(
        "cutoff_huang_rhys", "cutoff_n_vib", "validation_passed", "metadata_json",
    ))
    return {
        "level_rows": len(levels),
        "crossing_rows": len(crossings),
        "validation_rows": len(validation),
        "max_forbidden_coupling": max_forbidden,
        "crossings": crossings,
    }


def _audit_npz_finite(audit: Audit, path: Path, required: Sequence[str]) -> None:
    if not audit.check(path.is_file(), "NPZ exists", str(path)):
        return
    if not audit.check(path.stat().st_size > 0, "NPZ nonempty", str(path)):
        return
    try:
        with np.load(str(path), allow_pickle=False) as archive:
            missing = [key for key in required if key not in archive.files]
            audit.check(not missing, "NPZ key coverage", "{} missing {}".format(path, missing))
            bad: List[str] = []
            for key in archive.files:
                values = np.asarray(archive[key])
                if values.dtype.kind in "fc" and not np.all(np.isfinite(values)):
                    bad.append(key)
            audit.check(not bad, "NPZ numeric arrays finite", "{}: {}".format(path, bad))
    except (OSError, ValueError, KeyError) as error:
        audit.check(False, "NPZ readable", "{}: {}".format(path, error))


def _audit_sidebands_and_metrics(
    audit: Audit,
    summary: Sequence[Dict[str, str]],
    summary_by_id: Mapping[str, Dict[str, str]],
    accuracy_gate: Mapping[str, Any],
) -> Dict[str, Any]:
    numeric_columns = (
        "energy_00_eV", "omega0_meV", "huang_rhys", "lambda_meV", "j_meV",
        "j_sign", "r_abs", "r_signed", "temperature_K", "eta_meV",
        "eta_over_omega0", "analysis_x_min", "analysis_x_max", "internal_x_min",
        "internal_x_max", "dx", "analysis_samples", "n_vib", "runtime_ed_s",
        "runtime_ba_s", "runtime_scba_s", "ed_raw_area", "ba_raw_area",
        "scba_raw_area", "ba_raw_area_relative_difference",
        "scba_raw_area_relative_difference", "ba_l1", "scba_l1",
        "ba_w_over_omega0", "scba_w_over_omega0", "ed_peak_count",
        "ba_peak_count", "scba_peak_count", "ba_matched_count", "ba_missing_count",
        "ba_spurious_count", "ba_sideband_weight_error", "scba_matched_count",
        "scba_missing_count", "scba_spurious_count", "scba_sideband_weight_error",
        "ba_significant_missing_count", "ba_significant_spurious_count",
        "scba_significant_missing_count", "scba_significant_spurious_count",
        "significant_basin_weight_threshold", "tracked_exact_detuning_over_omega0",
        "tracked_bare_bright_overlap", "tracked_bare_overlap_margin",
        "tracked_continuity_overlap", "tracked_parity_expectation",
        "tracked_exact_pole_weight", "tracked_ed_peak_distance_over_omega0",
        "tracked_ed_basin_weight",
        "exact_projected_sigma_real_over_omega0",
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
    )
    _finite_columns(audit, summary, numeric_columns, "summary")

    # No-candidate tracking outcomes are represented by a coupled NaN triplet,
    # plus an explicit ambiguity reason and failed tracking gates.  Everywhere
    # else these three operational metrics must be finite.
    nullable_tracking = (
        "tracked_scba_shift_over_omega0",
        "tracked_scba_basin_weight",
        "tracked_scba_weight_error",
    )
    nullable_tracking_bad: List[str] = []
    undefined_tracking_count = 0
    for row in summary:
        finite = [np.isfinite(_float(row, column)) for column in nullable_tracking]
        if all(finite):
            continue
        if not any(finite):
            undefined_tracking_count += 1
            if (not row.get("tracked_scba_ambiguity", "")
                    or _truth(row.get("valid_tracked_shift"))
                    or _truth(row.get("valid_tracked_weight"))
                    or _truth(row.get("valid_tracking"))
                    or _truth(row.get("operational_validity_pass"))):
                nullable_tracking_bad.append(row.get("case_id", "") + ":invalid no-candidate flags")
        else:
            nullable_tracking_bad.append(row.get("case_id", "") + ":partial NaN triplet")
    audit.check(not nullable_tracking_bad, "nullable no-candidate tracking schema",
                ", ".join(nullable_tracking_bad[:12]))

    raw_max = max(abs(_float(row, "scba_raw_area_relative_difference")) for row in summary) if summary else float("inf")
    pole_max = max(abs(_float(row, "ed_pole_weight_error")) for row in summary) if summary else float("inf")
    common_max = max(abs(_float(row, "ed_common_omitted_weight")) for row in summary) if summary else float("inf")
    moment_max = max(abs(_float(row, "ed_first_moment_error_over_omega0")) for row in summary) if summary else float("inf")
    residual_abs_max = max(abs(_float(row, "scba_fixed_point_abs_residual_over_omega0")) for row in summary) if summary else float("inf")
    residual_rel_max = max(abs(_float(row, "scba_fixed_point_relative_residual")) for row in summary) if summary else float("inf")
    causality_max = max(_float(row, "scba_max_positive_im_sigma_over_omega0") for row in summary) if summary else float("inf")
    ed_min = min(_float(row, "ed_minimum_spectrum") for row in summary) if summary else -float("inf")
    scba_min = min(_float(row, "scba_minimum_spectrum") for row in summary) if summary else -float("inf")
    closure_max = max(abs(_float(row, "scba_channel_closure_over_omega0")) for row in summary) if summary else float("inf")

    audit.check(raw_max <= RAW_AREA_LIMIT, "all-case raw ED/SCBA area mismatch",
                "max {:.9e}, limit {:.1e}".format(raw_max, RAW_AREA_LIMIT))
    audit.check(pole_max <= ED_POLE_WEIGHT_LIMIT, "all-case ED pole-weight error",
                "max {:.9e}".format(pole_max))
    audit.check(common_max <= ED_POLE_WEIGHT_LIMIT, "all-case common-mode omitted weight",
                "max {:.9e}".format(common_max))
    audit.check(moment_max <= ED_MOMENT_LIMIT, "all-case ED first-moment error",
                "max {:.9e}".format(moment_max))
    audit.check(residual_abs_max <= SCBA_ABS_RESIDUAL_OVER_OMEGA_LIMIT,
                "all-case SCBA absolute residual/omega", "max {:.9e}".format(residual_abs_max))
    audit.check(residual_rel_max <= SCBA_REL_RESIDUAL_LIMIT,
                "all-case SCBA relative residual", "max {:.9e}".format(residual_rel_max))
    audit.check(causality_max <= SCBA_CAUSALITY_OVER_OMEGA_LIMIT,
                "all-case SCBA causality", "max {:.9e}".format(causality_max))
    audit.check(ed_min >= SPECTRUM_NEGATIVE_LIMIT, "all-case ED positivity",
                "min {:.9e}".format(ed_min))
    audit.check(scba_min >= SPECTRUM_NEGATIVE_LIMIT, "all-case SCBA positivity",
                "min {:.9e}".format(scba_min))
    audit.check(closure_max <= CHANNEL_CLOSURE_OVER_OMEGA_LIMIT,
                "all-case full-precision channel closure/omega", "max {:.9e}".format(closure_max))
    audit.check(all(
        0.0 <= _float(row, "scba_l1") <= 2.0 + 1.0e-10
        and _float(row, "scba_w_over_omega0") >= 0.0
        and _float(row, "ed_raw_area") > 0.0
        and _float(row, "scba_raw_area") > 0.0
        for row in summary
    ), "summary spectral metric ranges")

    sideband_columns = (
        "case_id", "family", "approximation", "status", "reference_weight",
        "candidate_weight", "absolute_weight_error",
    )
    sidebands = _load_csv(audit, RESULT_DIR / "sideband_matches.csv", sideband_columns)
    unknown_ids = sorted({row.get("case_id", "") for row in sidebands} - set(summary_by_id))
    audit.check(not unknown_ids, "sideband rows reference known cases", ", ".join(unknown_ids[:10]))
    valid_status = all(
        row.get("approximation") in ("BA", "SCBA")
        and row.get("status") in ("matched", "missing", "spurious")
        for row in sidebands
    )
    audit.check(valid_status, "sideband approximation/status values")

    counts: DefaultDict[Tuple[str, str], Counter] = defaultdict(Counter)
    significant: DefaultDict[Tuple[str, str], Counter] = defaultdict(Counter)
    absolute_errors: DefaultDict[Tuple[str, str], float] = defaultdict(float)
    reference_totals: DefaultDict[Tuple[str, str], float] = defaultdict(float)
    sideband_bad: List[str] = []
    for index, row in enumerate(sidebands, start=2):
        case_id = row.get("case_id", "")
        approximation = row.get("approximation", "")
        status = row.get("status", "")
        key = (case_id, approximation)
        counts[key][status] += 1
        ref_weight = _float(row, "reference_weight")
        cand_weight = _float(row, "candidate_weight")
        abs_error = _float(row, "absolute_weight_error")
        if not all(np.isfinite(value) for value in (ref_weight, cand_weight, abs_error)):
            sideband_bad.append("row {}".format(index))
            continue
        if status != "spurious":
            reference_totals[key] += ref_weight
        absolute_errors[key] += abs_error
        if status == "missing" and ref_weight >= SIGNIFICANT_WEIGHT - 1.0e-14:
            significant[key]["missing"] += 1
        if status == "spurious" and cand_weight >= SIGNIFICANT_WEIGHT - 1.0e-14:
            significant[key]["spurious"] += 1
    audit.check(not sideband_bad, "sideband weights finite", ", ".join(sideband_bad[:10]))

    topology_mismatch: List[str] = []
    aggregate_mismatch: List[str] = []
    for case_id, row in sorted(summary_by_id.items()):
        for approximation, prefix in (("BA", "ba"), ("SCBA", "scba")):
            key = (case_id, approximation)
            for status in ("matched", "missing", "spurious"):
                if _int(row, prefix + "_" + status + "_count") != counts[key][status]:
                    topology_mismatch.append(case_id + ":" + prefix + ":" + status)
            if _int(row, prefix + "_significant_missing_count") != significant[key]["missing"]:
                topology_mismatch.append(case_id + ":" + prefix + ":significant_missing")
            if _int(row, prefix + "_significant_spurious_count") != significant[key]["spurious"]:
                topology_mismatch.append(case_id + ":" + prefix + ":significant_spurious")
            total = reference_totals[key]
            computed = absolute_errors[key] / total if total > 0.0 else float("nan")
            if not _close(computed, _float(row, prefix + "_sideband_weight_error"), 2.0e-9):
                aggregate_mismatch.append(case_id + ":" + prefix)
    audit.check(not topology_mismatch, "sideband count recomputation",
                ", ".join(topology_mismatch[:15]))
    audit.check(not aggregate_mismatch, "sideband aggregate-weight recomputation",
                ", ".join(aggregate_mismatch[:15]))
    audit.check(all(_close(_float(row, "significant_basin_weight_threshold"), SIGNIFICANT_WEIGHT)
                    for row in summary), "significant basin threshold is 1%")

    l1_limit = float(accuracy_gate.get("L1_max", float("nan")))
    w_limit = float(accuracy_gate.get("W_over_omega0_max", float("nan")))
    shift_limit = float(accuracy_gate.get("tracked_shift_over_omega0_max", float("nan")))
    weight_limit = float(accuracy_gate.get("tracked_basin_weight_error_max", float("nan")))
    sideband_limit = float(accuracy_gate.get("aggregate_sideband_weight_error_max", float("nan")))
    raw_limit = float(accuracy_gate.get("raw_area_relative_error_max", float("nan")))
    audit.check(all(np.isfinite(value) for value in (
        l1_limit, w_limit, shift_limit, weight_limit, sideband_limit, raw_limit,
    )), "accuracy-gate thresholds finite")
    audit.check(_close(l1_limit, 0.10) and _close(w_limit, 0.01)
                and _close(shift_limit, 0.01) and _close(weight_limit, 0.05)
                and _close(sideband_limit, 0.10) and raw_limit <= RAW_AREA_LIMIT,
                "accuracy-gate declared values", str(dict(accuracy_gate)))

    flag_mismatch: List[str] = []
    pass_counts: Counter = Counter()
    for case_id, row in sorted(summary_by_id.items()):
        expected_flags = {
            "valid_l1": _float(row, "scba_l1") <= l1_limit,
            "valid_wasserstein": _float(row, "scba_w_over_omega0") <= w_limit,
            "valid_tracked_shift": abs(_float(row, "tracked_scba_shift_over_omega0")) <= shift_limit,
            "valid_tracked_weight": abs(_float(row, "tracked_scba_weight_error")) <= weight_limit,
            "valid_sideband_weight": _float(row, "scba_sideband_weight_error") <= sideband_limit,
            "valid_peak_topology": (
                significant[(case_id, "SCBA")]["missing"] == 0
                and significant[(case_id, "SCBA")]["spurious"] == 0
            ),
            "valid_raw_area": abs(_float(row, "scba_raw_area_relative_difference")) <= raw_limit,
            "valid_tracking": not row.get("tracked_ed_ambiguity", "") and not row.get("tracked_scba_ambiguity", ""),
        }
        expected_pass = all(expected_flags.values())
        for key, expected in expected_flags.items():
            if _truth(row.get(key)) != bool(expected):
                flag_mismatch.append(case_id + ":" + key)
        if _truth(row.get("operational_validity_pass")) != expected_pass:
            flag_mismatch.append(case_id + ":operational_validity_pass")
        if expected_pass:
            pass_counts[(row.get("family", ""), _float(row, "huang_rhys"), _int(row, "j_sign"))] += 1
    audit.check(not flag_mismatch, "operational gate independently recomputed",
                ", ".join(flag_mismatch[:20]))

    return {
        "sideband_rows": len(sidebands),
        "raw_max": raw_max,
        "pole_max": pole_max,
        "common_max": common_max,
        "moment_max": moment_max,
        "residual_abs_max": residual_abs_max,
        "residual_rel_max": residual_rel_max,
        "causality_max": causality_max,
        "ed_min": ed_min,
        "scba_min": scba_min,
        "closure_max": closure_max,
        "undefined_tracking_count": undefined_tracking_count,
        "pass_counts": pass_counts,
    }


def _audit_archives(
    audit: Audit,
    manifest: Sequence[Mapping[str, str]],
    summary_by_id: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    requested = [row for row in manifest if _truth(row.get("save_archive"))]
    audit.check(
        len(requested) == EXPECTED_ARCHIVE_COUNT,
        "requested spectrum-archive count",
        "observed {}, expected {}".format(len(requested), EXPECTED_ARCHIVE_COUNT),
    )
    expected_paths: Set[Path] = set()
    max_abs_closure = 0.0
    max_rel_closure = 0.0
    for row in requested:
        case_id = row.get("case_id", "")
        relative = row.get("archive_file", "")
        path = MODULE_DIR / relative if relative else Path("")
        audit.check(bool(relative), "requested archive path present", case_id)
        if not relative:
            continue
        expected_paths.add(path.resolve())
        if not audit.check(path.is_file(), "requested archive exists", str(path)):
            continue
        audit.check(path.stat().st_size > 0, "requested archive nonempty", str(path))
        try:
            with np.load(str(path), allow_pickle=False) as archive:
                missing = [key for key in EXPECTED_ARCHIVE_KEYS if key not in archive.files]
                audit.check(not missing, "archive key coverage", "{} missing {}".format(case_id, missing))
                if missing:
                    continue
                energy = np.asarray(archive["energy_eV"], dtype=float)
                mask = np.asarray(archive["analysis_mask"])
                audit.check(energy.ndim == 1 and energy.size >= 3 and np.all(np.diff(energy) > 0.0),
                            "archive energy grid", case_id)
                audit.check(mask.shape == energy.shape and mask.dtype.kind == "b" and np.count_nonzero(mask) >= 3,
                            "archive analysis mask", case_id)
                bad_arrays: List[str] = []
                for key in EXPECTED_ARCHIVE_KEYS:
                    if key in ("metadata_json", "analysis_mask", "energy_eV"):
                        continue
                    values = np.asarray(archive[key])
                    if values.shape != energy.shape or not np.all(np.isfinite(values)):
                        bad_arrays.append(key)
                audit.check(not bad_arrays, "archive array shapes/finiteness",
                            "{}: {}".format(case_id, bad_arrays))
                metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
                audit.check(metadata.get("schema_version") == SCHEMA_VERSION
                            and metadata.get("case_id") == case_id,
                            "archive metadata identity", case_id)
                model = metadata.get("model", {})
                audit.check(_close(float(model.get("omega0_meV", float("nan"))), _float(row, "omega0_meV"))
                            and _close(float(model.get("huang_rhys", float("nan"))), _float(row, "huang_rhys"))
                            and _close(float(model.get("j_meV", float("nan"))), _float(row, "j_meV")),
                            "archive model metadata", case_id)
                summary_meta = metadata.get("summary", {})
                audit.check(summary_meta.get("case_id") == case_id,
                            "archive summary metadata", case_id)

                sigma = np.asarray(archive["scba_sigma_eV"], dtype=np.complex128)
                channels = sum(
                    np.asarray(archive[key], dtype=np.complex128)
                    for key in (
                        "scba_sigma_from_bright_emission_eV",
                        "scba_sigma_from_bright_absorption_eV",
                        "scba_sigma_from_dark_emission_eV",
                        "scba_sigma_from_dark_absorption_eV",
                    )
                )
                residual = float(np.max(np.abs(channels - sigma)))
                scale = max(1.0e-30, float(np.max(np.abs(sigma))), float(np.max(np.abs(channels))))
                relative_residual = residual / scale
                max_abs_closure = max(max_abs_closure, residual)
                max_rel_closure = max(max_rel_closure, relative_residual)
                audit.check(relative_residual <= ARCHIVE_CHANNEL_CLOSURE_RELATIVE_LIMIT,
                            "archive complex64 channel closure", "{} relative {:.9e}".format(case_id, relative_residual))
        except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError) as error:
            audit.check(False, "archive readable", "{}: {}".format(case_id, error))

    actual_paths = {path.resolve() for path in (RESULT_DIR / "spectra").glob("*.npz")}
    audit.check(actual_paths == expected_paths, "archive inventory matches manifest",
                "expected {}, actual {}, orphan {}, missing {}".format(
                    len(expected_paths), len(actual_paths),
                    len(actual_paths - expected_paths), len(expected_paths - actual_paths),
                ))
    for row in manifest:
        if not _truth(row.get("save_archive")):
            audit.check(not row.get("archive_file", ""), "unrequested archive path blank", row.get("case_id", ""))
    return {
        "archive_count": len(expected_paths),
        "archive_max_abs_closure_eV": max_abs_closure,
        "archive_max_rel_closure": max_rel_closure,
    }


def _audit_wasserstein_line_scan_source(
    audit: Audit,
    summary_by_id: Mapping[str, Mapping[str, str]],
) -> int:
    """Trace the deliberately minimal, three-S line-plot source table."""

    path = RESULT_DIR / "wasserstein_line_scan_source.csv"
    rows = _load_csv(audit, path, LINE_SCAN_SOURCE_COLUMNS)
    _audit_exact_csv_schema(audit, path, LINE_SCAN_SOURCE_COLUMNS,
                            "Wasserstein line-scan source")
    expected_ids = {
        case_id for case_id, row in summary_by_id.items()
        if row.get("family") == "primary_fixed_S"
        and any(_close(_float(row, "huang_rhys"), value)
                for value in SELECTED_LINE_SCAN_S)
    }
    expected_count = len(SELECTED_LINE_SCAN_S) * 2 * EXPECTED_PRIMARY_R_POINTS
    source_ids = [row.get("case_id", "") for row in rows]
    audit.check(len(rows) == expected_count, "Wasserstein line-scan source row count",
                "observed {}, expected {}".format(len(rows), expected_count))
    audit.check(all(source_ids) and len(set(source_ids)) == len(source_ids),
                "Wasserstein line-scan source IDs nonempty and unique")
    audit.check(set(source_ids) == expected_ids,
                "Wasserstein line-scan source selected-S coverage",
                "missing {}, extra {}".format(
                    len(expected_ids - set(source_ids)), len(set(source_ids) - expected_ids)))
    _finite_columns(audit, rows, LINE_SCAN_SOURCE_COLUMNS[1:],
                    "Wasserstein line-scan source")

    expected_order = sorted(expected_ids, key=lambda case_id: (
        _int(summary_by_id[case_id], "j_sign"),
        _float(summary_by_id[case_id], "huang_rhys"),
        _float(summary_by_id[case_id], "r_abs"),
    ))
    audit.check(source_ids == expected_order, "Wasserstein line-scan source row order")
    mismatch: List[str] = []
    for row in rows:
        case_id = row.get("case_id", "")
        source = summary_by_id.get(case_id)
        if source is None:
            mismatch.append(case_id + ":missing-source")
            continue
        for column in LINE_SCAN_SOURCE_COLUMNS:
            same = (row.get(column, "") == source.get(column, "")
                    if column == "case_id" else
                    _close(_float(row, column), _float(source, column), 2.0e-12))
            if not same:
                mismatch.append(case_id + ":" + column)
                break
    audit.check(not mismatch, "Wasserstein line-scan source exactly traces summary",
                ", ".join(mismatch[:15]))

    selected_baselines = [
        row for row in summary_by_id.values()
        if row.get("family") == "primary_J0_baseline"
        and any(_close(_float(row, "huang_rhys"), value)
                for value in SELECTED_LINE_SCAN_S)
    ]
    selected_baselines.sort(key=lambda row: _float(row, "huang_rhys"))
    audit.check(
        len(selected_baselines) == len(SELECTED_LINE_SCAN_S)
        and all(_close(_float(row, "huang_rhys"), s_value)
                for row, s_value in zip(selected_baselines, SELECTED_LINE_SCAN_S)),
        "selected J0 baseline S coverage",
    )
    audit.check(
        all(_int(row, "j_sign") == 0
            and _close(_float(row, "j_meV"), 0.0)
            and _close(_float(row, "r_abs"), 0.0)
            and np.isfinite(_float(row, "scba_w_over_omega0"))
            for row in selected_baselines),
        "selected line-scan baselines are finite J0 references",
    )
    return len(rows)


def _audit_legacy_frequency_map_source(
    audit: Audit,
    summary_by_id: Mapping[str, Mapping[str, str]],
) -> int:
    """Keep the frozen command-14 map source traceable after mesh refinement."""

    path = RESULT_DIR / "frequency_resonance_map_source.csv"
    rows = _load_csv(audit, path, LEGACY_FREQUENCY_MAP_COLUMNS)
    _audit_exact_csv_schema(audit, path, LEGACY_FREQUENCY_MAP_COLUMNS,
                            "legacy frequency-map source")
    legacy_r = {round(value, 12) for value in _legacy_primary_r_grid()}
    audit.check(len(legacy_r) == LEGACY_PRIMARY_R_POINTS,
                "internal legacy primary r-grid definition")
    expected_ids = {
        case_id for case_id, source in summary_by_id.items()
        if source.get("family") == "primary_fixed_S"
        and round(_float(source, "r_abs"), 12) in legacy_r
    }
    expected_count = len(PRIMARY_S) * 2 * LEGACY_PRIMARY_R_POINTS
    source_ids = [row.get("case_id", "") for row in rows]
    audit.check(len(rows) == expected_count, "legacy frequency-map source row count",
                "observed {}, expected {}".format(len(rows), expected_count))
    audit.check(all(source_ids) and len(set(source_ids)) == len(source_ids),
                "legacy frequency-map source IDs nonempty and unique")
    audit.check(set(source_ids) == expected_ids,
                "legacy frequency-map source retained-grid coverage",
                "missing {}, extra {}".format(
                    len(expected_ids - set(source_ids)), len(set(source_ids) - expected_ids)))
    string_columns = {
        "case_id", "valid_wasserstein", "valid_sideband_weight",
        "valid_peak_topology", "operational_validity_pass",
    }
    mismatch: List[str] = []
    for row in rows:
        case_id = row.get("case_id", "")
        source = summary_by_id.get(case_id)
        if source is None:
            mismatch.append(case_id + ":missing-source")
            continue
        for column in LEGACY_FREQUENCY_MAP_COLUMNS:
            same = (row.get(column, "") == source.get(column, "")
                    if column in string_columns else
                    _close(_float(row, column), _float(source, column), 2.0e-12))
            if not same:
                mismatch.append(case_id + ":" + column)
                break
    audit.check(not mismatch, "legacy frequency-map source exactly traces summary",
                ", ".join(mismatch[:15]))
    return len(rows)


def _audit_legacy_representative_table(
    audit: Audit,
    manifest: Sequence[Mapping[str, str]],
    summary_by_id: Mapping[str, Mapping[str, str]],
) -> int:
    """Retain the six-panel command-14 source table as audited raw data."""

    path = RESULT_DIR / "representative_summary_table.csv"
    rows = _load_csv(audit, path, LEGACY_REPRESENTATIVE_COLUMNS)
    _audit_exact_csv_schema(audit, path, LEGACY_REPRESENTATIVE_COLUMNS,
                            "legacy representative table")
    table_ids = [row.get("case_id", "") for row in rows]
    archived_ids = {
        row.get("case_id", "") for row in manifest if _truth(row.get("save_archive"))
    }
    audit.check(len(rows) == 6, "legacy representative-table row count",
                "observed {}, expected 6".format(len(rows)))
    audit.check(all(table_ids) and len(set(table_ids)) == len(table_ids),
                "legacy representative-table case IDs nonempty and unique")
    audit.check(set(table_ids) <= archived_ids,
                "legacy representative-table cases retain production archives")
    audit.check([_int(row, "display_order") for row in rows] == list(range(1, 7)),
                "legacy representative-table display order")
    audit.check(len({row.get("panel", "") for row in rows}) == len(rows)
                and all(row.get("panel", "") for row in rows),
                "legacy representative-table panels nonempty and unique")
    audit.check(all(row.get("role", "").strip() for row in rows),
                "legacy representative-table roles nonempty")

    numeric_columns = tuple(
        column for column in LEGACY_REPRESENTATIVE_COLUMNS
        if column not in ("display_order", "panel", "case_id", "role",
                          "tracked_ed_ambiguity", "tracked_scba_ambiguity",
                          "operational_validity_pass", "archive_file")
    )
    mismatch: List[str] = []
    for row in rows:
        case_id = row.get("case_id", "")
        source = summary_by_id.get(case_id)
        if source is None:
            mismatch.append(case_id + ":missing-source")
            continue
        for column in numeric_columns:
            table_value = _float(row, column)
            source_value = _float(source, column)
            if not ((math.isnan(table_value) and math.isnan(source_value))
                    or _close(table_value, source_value, 2.0e-12)):
                mismatch.append(case_id + ":" + column)
                break
        for column in ("tracked_ed_ambiguity", "tracked_scba_ambiguity",
                       "operational_validity_pass", "archive_file"):
            if row.get(column, "") != source.get(column, ""):
                mismatch.append(case_id + ":" + column)
                break
    audit.check(not mismatch, "legacy representative table exactly traces summary",
                ", ".join(mismatch[:15]))
    return len(rows)


def _audit_peak_manifest(
    audit: Audit,
    summary_by_id: Mapping[str, Mapping[str, str]],
) -> Dict[str, Any]:
    """Recompute every reported extremum from measured scan points."""

    path = RESULT_DIR / "wasserstein_peak_manifest.csv"
    rows = _load_csv(audit, path, PEAK_MANIFEST_COLUMNS)
    _audit_exact_csv_schema(audit, path, PEAK_MANIFEST_COLUMNS,
                            "Wasserstein peak manifest")
    expected_keys = {
        (round(s_value, 12), sign, guide)
        for s_value in SELECTED_LINE_SCAN_S
        for sign in (-1, 1)
        for guide in INTEGER_GUIDES
    }
    keys = [
        (round(_float(row, "huang_rhys"), 12), _int(row, "j_sign"),
         _int(row, "integer_guide"))
        for row in rows
    ]
    audit.check(len(rows) == len(expected_keys), "Wasserstein peak-manifest row count",
                "observed {}, expected {}".format(len(rows), len(expected_keys)))
    audit.check(set(keys) == expected_keys and len(keys) == len(set(keys)),
                "Wasserstein peak-manifest S/sign/guide coverage")
    _finite_columns(
        audit, rows,
        ("huang_rhys", "j_sign", "integer_guide", "r_peak", "w_peak",
         "r_reference_min", "w_reference_min"),
        "Wasserstein peak manifest",
    )

    primary_groups: DefaultDict[Tuple[float, int], List[Mapping[str, str]]] = defaultdict(list)
    for source in summary_by_id.values():
        if (source.get("family") == "primary_fixed_S"
                and any(_close(_float(source, "huang_rhys"), value)
                        for value in SELECTED_LINE_SCAN_S)):
            primary_groups[(round(_float(source, "huang_rhys"), 12),
                            _int(source, "j_sign"))].append(source)
    for values in primary_groups.values():
        values.sort(key=lambda row: _float(row, "r_abs"))

    mismatch: List[str] = []
    status_counts: Counter = Counter()
    by_key: Dict[Tuple[float, int, int], Mapping[str, str]] = {}
    for row, key in zip(rows, keys):
        by_key[key] = row
        s_value, sign, guide = key
        label = "odd allowed guide" if guide % 2 else "even parity-forbidden control"
        values = primary_groups.get((s_value, sign), [])
        positions = [_float(source, "r_abs") for source in values]
        core = [source for source in values
                if abs(_float(source, "r_abs") - guide) <= 0.15 + 1.0e-12]
        reference = [source for source in values
                     if 0.15 + 1.0e-12 < abs(_float(source, "r_abs") - guide)
                     <= 0.50 + 1.0e-12]
        local_maxima = [
            values[index] for index in range(1, len(values) - 1)
            if abs(positions[index] - guide) < 0.50 - 1.0e-12
            and _float(values[index], "scba_w_over_omega0")
            > _float(values[index - 1], "scba_w_over_omega0")
            and _float(values[index], "scba_w_over_omega0")
            > _float(values[index + 1], "scba_w_over_omega0")
        ]
        if not core or not reference:
            mismatch.append("S={} sign={} n={}:incomplete-source-window".format(
                s_value, sign, guide))
            continue
        expected_peak = max(
            local_maxima if local_maxima else core,
            key=lambda source: _float(source, "scba_w_over_omega0"),
        )
        expected_reference = min(
            reference, key=lambda source: _float(source, "scba_w_over_omega0"))
        expected_status = (
            "resolved local maximum" if local_maxima
            else "guide-window maximum; no resolved local cusp"
        )
        status_counts[(sign, expected_status)] += 1
        comparisons = (
            (row.get("feature_label", "") == label, "feature-label"),
            (row.get("extremum_type", "") == expected_status, "extremum-status"),
            (row.get("peak_case_id", "") == expected_peak.get("case_id", ""), "peak-ID"),
            (_close(_float(row, "r_peak"), _float(expected_peak, "r_abs"), 2.0e-12),
             "peak-r"),
            (_close(_float(row, "w_peak"),
                    _float(expected_peak, "scba_w_over_omega0"), 2.0e-12), "peak-w"),
            (row.get("reference_case_id", "") == expected_reference.get("case_id", ""),
             "reference-ID"),
            (_close(_float(row, "r_reference_min"),
                    _float(expected_reference, "r_abs"), 2.0e-12), "reference-r"),
            (_close(_float(row, "w_reference_min"),
                    _float(expected_reference, "scba_w_over_omega0"), 2.0e-12),
             "reference-w"),
            (row.get("reference_label", "") == "noninteger background", "reference-label"),
            (row.get("search_window", "")
             == "|r-n| < 0.50; strict local maxima preferred", "search-window"),
            (row.get("reference_window", "") == "0.15 < |r-n| <= 0.50",
             "reference-window"),
        )
        failure = next((name for passed, name in comparisons if not passed), None)
        if failure:
            mismatch.append("S={} sign={} n={}:{}".format(s_value, sign, guide, failure))
    audit.check(not mismatch, "Wasserstein peak manifest recomputes from summary",
                ", ".join(mismatch[:18]))
    return {"rows": len(rows), "by_key": by_key, "status_counts": status_counts}


def _audit_representative_table(
    audit: Audit,
    manifest: Sequence[Mapping[str, str]],
    summary_by_id: Mapping[str, Mapping[str, str]],
    peaks: Mapping[str, Any],
) -> Tuple[int, Set[str]]:
    """Verify the four Wasserstein-selected spectra and their compact schema."""

    path = RESULT_DIR / "representative_spectra_manifest.csv"
    rows = _load_csv(audit, path, REPRESENTATIVE_COLUMNS)
    _audit_exact_csv_schema(audit, path, REPRESENTATIVE_COLUMNS,
                            "representative-spectra manifest")
    table_ids = [row.get("case_id", "") for row in rows]
    archived_ids = {
        row.get("case_id", "") for row in manifest if _truth(row.get("save_archive"))
    }
    audit.check(len(rows) == 4, "representative-spectra row count",
                "observed {}, expected 4".format(len(rows)))
    audit.check(all(table_ids) and len(set(table_ids)) == len(table_ids),
                "representative-spectra case IDs nonempty and unique")
    audit.check(set(table_ids) <= archived_ids,
                "representative-spectra cases have production archives")
    audit.check([_int(row, "display_order") for row in rows] == [1, 2, 3, 4],
                "representative-spectra display order")
    audit.check([row.get("panel", "") for row in rows] == ["a", "b", "c", "d"],
                "representative-spectra panel order")

    peak_lookup = peaks.get("by_key", {})
    guide_one = peak_lookup.get((0.05, 1, 1), {})
    guide_two = peak_lookup.get((0.05, 1, 2), {})
    positive_peak_r = _float(guide_one, "r_peak")
    negative_counterpart = next((
        case_id for case_id, source in summary_by_id.items()
        if source.get("family") == "primary_fixed_S"
        and _close(_float(source, "huang_rhys"), 0.05)
        and _int(source, "j_sign") == -1
        and _close(_float(source, "r_abs"), positive_peak_r, 2.0e-12)
    ), "")
    expected_by_role = {
        "off-integer background": guide_one.get("reference_case_id", ""),
        "first odd allowed feature": guide_one.get("peak_case_id", ""),
        "even parity-forbidden control": guide_two.get("peak_case_id", ""),
        "negative-$J$ counterpart": negative_counterpart,
    }
    observed_by_role = {row.get("role", ""): row.get("case_id", "") for row in rows}
    audit.check(observed_by_role == expected_by_role,
                "representative spectra follow measured peak manifest",
                "observed {}, expected {}".format(observed_by_role, expected_by_role))

    numeric_columns = tuple(
        column for column in REPRESENTATIVE_COLUMNS
        if column not in ("display_order", "panel", "case_id", "role", "archive_file")
    )
    _finite_columns(audit, rows, numeric_columns, "representative-spectra manifest")
    mismatch: List[str] = []
    for row in rows:
        case_id = row.get("case_id", "")
        source = summary_by_id.get(case_id)
        if source is None:
            mismatch.append(case_id + ":missing-source")
            continue
        for column in numeric_columns:
            if not _close(_float(row, column), _float(source, column), 2.0e-12):
                mismatch.append(case_id + ":" + column)
                break
        if row.get("archive_file", "") != source.get("archive_file", ""):
            mismatch.append(case_id + ":archive-file")
        archive_text = row.get("archive_file", "")
        archive = Path(archive_text)
        if archive_text and not (archive if archive.is_absolute() else MODULE_DIR / archive).is_file():
            mismatch.append(case_id + ":archive-missing")
    audit.check(not mismatch,
                "representative-spectra manifest exactly traces summary/archive data",
                ", ".join(mismatch[:15]))
    return len(rows), set(table_ids)


def _audit_resonance_sweep_table(
    audit: Audit,
    manifest: Sequence[Mapping[str, str]],
    summary_by_id: Mapping[str, Mapping[str, str]],
    peaks: Mapping[str, Any],
) -> Tuple[int, Set[str]]:
    """Trace all eight command-16 first-resonance spectral cuts."""

    path = RESULT_DIR / "resonance_sweep_spectra_manifest.csv"
    rows = _load_csv(audit, path, RESONANCE_SWEEP_COLUMNS)
    _audit_exact_csv_schema(audit, path, RESONANCE_SWEEP_COLUMNS,
                            "resonance-sweep spectra manifest")
    audit.check(len(rows) == len(EXPECTED_RESONANCE_SWEEP),
                "resonance-sweep manifest row count",
                "observed {}, expected {}".format(
                    len(rows), len(EXPECTED_RESONANCE_SWEEP)))
    audit.check([_int(row, "display_order") for row in rows]
                == list(range(1, len(EXPECTED_RESONANCE_SWEEP) + 1)),
                "resonance-sweep display order")

    expected_ids: List[str] = []
    expected_by_position: List[Mapping[str, str]] = []
    lookup_mismatch: List[str] = []
    for s_value, r_value, _, _, _ in EXPECTED_RESONANCE_SWEEP:
        matches = [
            source for source in summary_by_id.values()
            if source.get("family") == "primary_fixed_S"
            and _close(_float(source, "huang_rhys"), s_value)
            and _int(source, "j_sign") == 1
            and _close(_float(source, "r_abs"), r_value, 2.0e-12)
        ]
        if len(matches) != 1:
            lookup_mismatch.append(
                "S={} r={}:{} summary matches".format(
                    s_value, r_value, len(matches)))
            expected_by_position.append({})
            expected_ids.append("")
        else:
            expected_by_position.append(matches[0])
            expected_ids.append(matches[0].get("case_id", ""))
    audit.check(not lookup_mismatch, "resonance-sweep source-point coverage",
                ", ".join(lookup_mismatch))

    observed_ids = [row.get("case_id", "") for row in rows]
    audit.check(observed_ids == expected_ids,
                "resonance-sweep ordered case selection",
                "observed {}, expected {}".format(observed_ids, expected_ids))
    audit.check(all(observed_ids) and len(set(observed_ids)) == len(observed_ids),
                "resonance-sweep case IDs nonempty and unique")

    expected_roles = [item[4] for item in EXPECTED_RESONANCE_SWEEP]
    expected_panels = [item[2] for item in EXPECTED_RESONANCE_SWEEP]
    expected_labels = [item[3] for item in EXPECTED_RESONANCE_SWEEP]
    audit.check([row.get("role", "") for row in rows] == expected_roles,
                "resonance-sweep physical roles")
    audit.check([row.get("figure_panel", "") for row in rows] == expected_panels,
                "resonance-sweep figure panels")
    audit.check([row.get("cut_label", "") for row in rows] == expected_labels,
                "resonance-sweep cut labels")
    audit.check(all(row.get("selection_rule", "").strip() for row in rows),
                "resonance-sweep selection rules nonempty")

    archived_ids = {
        row.get("case_id", "") for row in manifest if _truth(row.get("save_archive"))
    }
    audit.check(set(observed_ids) <= archived_ids,
                "resonance-sweep cases have requested production archives")

    peak_lookup = peaks.get("by_key", {})
    peak_selection_ok = True
    for s_value, peak_r, reference_r in ((0.05, 0.88, 1.45),
                                         (0.10, 0.80, 1.35)):
        peak_row = peak_lookup.get((s_value, 1, 1), {})
        peak_selection_ok = peak_selection_ok and (
            _close(_float(peak_row, "r_peak"), peak_r, 2.0e-12)
            and _close(_float(peak_row, "r_reference_min"), reference_r, 2.0e-12)
            and peak_row.get("extremum_type", "") == "resolved local maximum"
        )
    audit.check(peak_selection_ok,
                "resonance-sweep maxima/references trace peak manifest")

    baseline_by_s = {
        round(_float(source, "huang_rhys"), 12): source
        for source in summary_by_id.values()
        if source.get("family") == "primary_J0_baseline"
    }
    numeric_source_columns = (
        "huang_rhys", "j_sign", "j_meV", "r_abs", "omega0_meV",
        "eta_over_omega0", "scba_w_over_omega0", "ed_raw_area",
        "scba_raw_area", "scba_raw_area_relative_difference",
    )
    _finite_columns(
        audit, rows,
        numeric_source_columns + ("local_baseline_w_over_omega0",),
        "resonance-sweep spectra manifest",
    )
    mismatch: List[str] = []
    for row, source in zip(rows, expected_by_position):
        case_id = row.get("case_id", "")
        if not source:
            continue
        for column in numeric_source_columns:
            if not _close(_float(row, column), _float(source, column), 2.0e-12):
                mismatch.append(case_id + ":" + column)
                break
        baseline = baseline_by_s.get(round(_float(row, "huang_rhys"), 12), {})
        if not _close(
                _float(row, "local_baseline_w_over_omega0"),
                _float(baseline, "scba_w_over_omega0"), 2.0e-12):
            mismatch.append(case_id + ":local-baseline")
        if row.get("archive_file", "") != source.get("archive_file", ""):
            mismatch.append(case_id + ":archive-file")
        archive_text = row.get("archive_file", "")
        archive = Path(archive_text)
        archive_path = archive if archive.is_absolute() else MODULE_DIR / archive
        if not archive_text or not archive_path.is_file():
            mismatch.append(case_id + ":archive-missing")
        if abs(_float(row, "scba_raw_area_relative_difference")) > RAW_AREA_LIMIT:
            mismatch.append(case_id + ":raw-area")
    audit.check(not mismatch,
                "resonance-sweep manifest exactly traces summary/baseline/archives",
                ", ".join(mismatch[:20]))
    return len(rows), set(observed_ids)


def _audit_generated_validation(audit: Audit) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    validation = _read_json(audit, VALIDATION_DIR / "validation.json")
    audit.check(validation.get("schema_version") == SCHEMA_VERSION, "validation schema version")
    checks = validation.get("checks", [])
    audit.check(isinstance(checks, list) and bool(checks), "generated validation checks present")
    if isinstance(checks, list):
        audit.check(all(isinstance(item, dict) and item.get("pass") is True for item in checks),
                    "all generated validation checks pass",
                    ", ".join(str(item.get("name")) for item in checks
                              if isinstance(item, dict) and item.get("pass") is not True)[:500])
    audit.check(validation.get("pass") is True, "generated validation overall pass")
    accuracy_gate = validation.get("accuracy_gate", {})
    audit.check(isinstance(accuracy_gate, dict), "accuracy gate present in validation JSON")

    required_csvs = {
        "legacy_reproduction.csv": (6, ("huang_rhys", "l1_abs_difference", "w_over_omega_abs_difference")),
        "dimensionless_scaling.csv": (4, ("huang_rhys", "r_abs", "j_sign", "x_grid_max_abs",
                                                  "dimensionless_ed_green_max_abs",
                                                  "dimensionless_ed_green_max_relative",
                                                  "dimensionless_ed_spectrum_l1",
                                                  "dimensionless_ed_spectrum_w_over_omega0",
                                                  "dimensionless_scba_green_max_abs",
                                                  "dimensionless_scba_green_max_relative",
                                                  "dimensionless_scba_spectrum_l1",
                                                  "dimensionless_scba_spectrum_w_over_omega0")),
        "j0_scaling.csv": (14, ("huang_rhys", "reference_omega0_meV", "candidate_omega0_meV",
                                         "x_grid_max_abs", "dimensionless_ed_green_max_abs",
                                         "dimensionless_ed_green_max_relative",
                                         "dimensionless_ed_spectrum_l1",
                                         "dimensionless_ed_spectrum_w_over_omega0",
                                         "dimensionless_scba_green_max_abs",
                                         "dimensionless_scba_green_max_relative",
                                         "dimensionless_scba_spectrum_l1",
                                         "dimensionless_scba_spectrum_w_over_omega0")),
        "grid_padding_convergence.csv": (10, ("case_label", "variant", "huang_rhys", "r_signed",
                                                       "omega0_meV", "ed_solution_l1_change",
                                                       "scba_solution_l1_change",
                                                       "scba_vs_ed_l1_change",
                                                       "scba_vs_ed_w_over_omega0_change")),
        "poisson_validation.csv": (3, ("huang_rhys", "l1", "w_over_omega0")),
        "parity_validation.csv": (3, ("huang_rhys", "commutator_over_omega0",
                                              "opposite_parity_hamiltonian_block_over_omega0",
                                              "eigenstate_parity_leakage")),
    }
    inventory: Dict[str, Any] = {}
    for filename, specification in required_csvs.items():
        expected_rows, columns = specification
        rows = _load_csv(audit, VALIDATION_DIR / filename, columns)
        audit.check(len(rows) == expected_rows, filename + " row count",
                    "observed {}, expected {}".format(len(rows), expected_rows))
        numeric = [column for column in columns if column not in ("case_label", "variant")]
        _finite_columns(audit, rows, numeric, filename)
        inventory[filename] = rows

    grid_rows = inventory.get("grid_padding_convergence.csv", [])
    audit.check(Counter(row.get("variant", "") for row in grid_rows)
                == {"dx_half": 5, "padding_wide": 5},
                "grid/padding validation variant inventory")
    j0_rows = inventory.get("j0_scaling.csv", [])
    audit.check({ _float(row, "huang_rhys") for row in j0_rows } == set(PRIMARY_S),
                "J0 scaling covers all seven S values")
    audit.check(Counter(_float(row, "candidate_omega0_meV") for row in j0_rows)
                == {10.0: 7, 200.0: 7}, "J0 scaling frequency inventory")

    # Independently apply the published tolerances to the source CSVs instead
    # of trusting only the pass booleans in validation.json.
    def maximum_absolute(rows: Sequence[Mapping[str, str]], column: str) -> float:
        return max((abs(_float(row, column)) for row in rows), default=float("inf"))

    for filename in ("dimensionless_scaling.csv", "j0_scaling.csv"):
        rows = inventory.get(filename, [])
        limits = {
            "x_grid_max_abs": 2.0e-12,
            "dimensionless_ed_green_max_abs": 3.0e-6,
            "dimensionless_ed_green_max_relative": 1.0e-7,
            "dimensionless_ed_spectrum_l1": 1.0e-5,
            "dimensionless_ed_spectrum_w_over_omega0": 1.0e-6,
            "dimensionless_scba_green_max_abs": 2.0e-10,
            "dimensionless_scba_green_max_relative": 1.0e-7,
            "dimensionless_scba_spectrum_l1": 1.0e-5,
            "dimensionless_scba_spectrum_w_over_omega0": 1.0e-6,
        }
        scaling_bad = [
            "{}={:.9e}>{:.1e}".format(column, maximum_absolute(rows, column), limit)
            for column, limit in limits.items()
            if maximum_absolute(rows, column) > limit
        ]
        audit.check(not scaling_bad, filename + " independently satisfies scaling limits",
                    ", ".join(scaling_bad))

    grid_limits = {
        "dx_half": {
            "ed_solution_l1_change": 5.0e-3,
            "scba_solution_l1_change": 5.0e-3,
            "scba_vs_ed_l1_change": 5.0e-3,
            "scba_vs_ed_w_over_omega0_change": 5.0e-4,
        },
        "padding_wide": {
            "ed_solution_l1_change": 5.0e-5,
            "scba_solution_l1_change": 5.0e-5,
            "scba_vs_ed_l1_change": 5.0e-5,
            "scba_vs_ed_w_over_omega0_change": 5.0e-6,
        },
    }
    grid_bad: List[str] = []
    for variant, limits in grid_limits.items():
        subset = [row for row in grid_rows if row.get("variant") == variant]
        for column, limit in limits.items():
            observed = maximum_absolute(subset, column)
            if observed > limit:
                grid_bad.append("{}:{}={:.9e}>{:.1e}".format(
                    variant, column, observed, limit))
    audit.check(not grid_bad, "grid/padding CSV independently satisfies limits",
                ", ".join(grid_bad))

    legacy_rows = inventory.get("legacy_reproduction.csv", [])
    audit.check(maximum_absolute(legacy_rows, "l1_abs_difference") <= 3.0e-4
                and maximum_absolute(legacy_rows, "w_over_omega_abs_difference") <= 3.0e-4,
                "legacy reproduction CSV independently satisfies limits")
    poisson_rows = inventory.get("poisson_validation.csv", [])
    audit.check(maximum_absolute(poisson_rows, "l1") <= 2.0e-4
                and maximum_absolute(poisson_rows, "w_over_omega0") <= 2.0e-5,
                "Poisson CSV independently satisfies limits")
    parity_rows = inventory.get("parity_validation.csv", [])
    audit.check(maximum_absolute(parity_rows, "commutator_over_omega0") <= 2.0e-13
                and maximum_absolute(parity_rows, "opposite_parity_hamiltonian_block_over_omega0") <= 2.0e-13
                and maximum_absolute(parity_rows, "eigenstate_parity_leakage") <= 2.0e-10,
                "parity CSV independently satisfies limits")

    text_path = VALIDATION_DIR / "numerical_validation.txt"
    if audit.check(text_path.is_file(), "numerical validation text exists", str(text_path)):
        try:
            text_value = text_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            audit.check(False, "numerical validation text readable", str(error))
        else:
            audit.check(bool(text_value.strip()), "numerical validation text nonempty")
            audit.check("overall=PASS" in text_value, "numerical validation text reports PASS")
    return accuracy_gate if isinstance(accuracy_gate, dict) else {}, inventory


def _source_file_valid(audit: Audit, path: Path) -> bool:
    if not audit.check(path.is_file(), "figure source exists", str(path)):
        return False
    if not audit.check(path.stat().st_size > 0, "figure source nonempty", str(path)):
        return False
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows = _load_csv(audit, path)
        return bool(rows)
    if suffix == ".npz":
        _audit_npz_finite(audit, path, required=())
        return True
    if suffix == ".json":
        return bool(_read_json(audit, path))
    audit.check(False, "supported figure-source format", str(path))
    return False


def _audit_png(audit: Audit, path: Path) -> None:
    audit.check(path.stat().st_size >= 1000, "PNG nontrivial size", str(path))
    checked = False
    try:
        from PIL import Image  # type: ignore
    except ImportError:
        pass
    else:
        try:
            with Image.open(str(path)) as image:
                width, height = image.size
                image.verify()
            audit.check(width >= 100 and height >= 100, "PNG readable dimensions", str(path))
            checked = True
        except Exception as error:  # Pillow raises several format-specific exceptions.
            audit.check(False, "PNG readable with Pillow", "{}: {}".format(path, error))
            checked = True
    if not checked:
        try:
            import matplotlib.image as mpimg  # type: ignore
        except ImportError:
            signature = path.read_bytes()[:8]
            audit.check(signature == b"\x89PNG\r\n\x1a\n", "PNG signature", str(path))
        else:
            try:
                image = np.asarray(mpimg.imread(str(path)))
                audit.check(image.ndim in (2, 3) and image.shape[0] >= 100 and image.shape[1] >= 100
                            and np.all(np.isfinite(image)), "PNG readable with matplotlib", str(path))
            except Exception as error:
                audit.check(False, "PNG readable with matplotlib", "{}: {}".format(path, error))


def _audit_figures(audit: Audit, sweep_ids: Set[str]) -> Dict[str, Any]:
    audit.check(FIGURE_DIR.is_dir(), "frequency-scan figure directory exists", str(FIGURE_DIR))
    pngs = sorted(FIGURE_DIR.glob("*.png")) if FIGURE_DIR.is_dir() else []
    pdfs = sorted(FIGURE_DIR.glob("*.pdf")) if FIGURE_DIR.is_dir() else []
    png_stems = {path.stem for path in pngs}
    pdf_stems = {path.stem for path in pdfs}
    expected_stems = (ACTIVE_FIGURE_STEMS | COMMAND15_FIGURE_STEMS
                      | COMMAND14_FIGURE_STEMS)
    audit.check(png_stems == expected_stems, "active and preserved PNG figure coexistence",
                "observed {}, expected {}".format(sorted(png_stems), sorted(expected_stems)))
    audit.check(pdf_stems == expected_stems, "active and preserved PDF figure coexistence",
                "observed {}, expected {}".format(sorted(pdf_stems), sorted(expected_stems)))
    for path in pngs:
        _audit_png(audit, path)
    for path in pdfs:
        audit.check(path.stat().st_size >= 1000, "PDF nontrivial size", str(path))
        try:
            header = path.read_bytes()[:5]
        except OSError as error:
            audit.check(False, "PDF readable", "{}: {}".format(path, error))
        else:
            audit.check(header == b"%PDF-", "PDF header", str(path))

    source_manifest_path = FIGURE_DIR / "figure_source_manifest.csv"
    traced_stems: Set[str] = set()
    source_count = 0
    source_rows = _load_csv(audit, source_manifest_path, SOURCE_MANIFEST_COLUMNS)
    _audit_exact_csv_schema(audit, source_manifest_path, SOURCE_MANIFEST_COLUMNS,
                            "figure source manifest")
    audit.check(len(source_rows) == 13, "figure source-manifest row count",
                "observed {}, expected 13".format(len(source_rows)))
    declared_stems = {row.get("figure_stem", "") for row in source_rows}
    audit.check(declared_stems == ACTIVE_FIGURE_STEMS,
                "figure source-manifest active command-16 stem coverage",
                "declared {}, required {}".format(
                    sorted(declared_stems), sorted(ACTIVE_FIGURE_STEMS)))
    source_rows_by_stem = Counter(row.get("figure_stem", "") for row in source_rows)
    audit.check(source_rows_by_stem == {
        "wasserstein_line_scan": 3,
        "resonance_sweep_spectra": 10,
    }, "figure source-manifest per-figure inventory",
                "observed {}".format(dict(source_rows_by_stem)))
    audit.check(all(row.get(column, "").strip() for row in source_rows
                    for column in SOURCE_MANIFEST_COLUMNS[:-1]),
                "figure source-manifest required values nonempty")
    audit.check(len({(row.get("figure_stem", ""), row.get("source_file", ""))
                     for row in source_rows}) == len(source_rows),
                "figure source-manifest rows unique")

    source_names: DefaultDict[str, Set[str]] = defaultdict(set)
    spectral_selections: Set[str] = set()
    for row in source_rows:
        figure_name = Path(row.get("figure_stem", "")).stem
        source_text = row.get("source_file", "")
        if not source_text:
            continue
        candidate = Path(source_text)
        if not candidate.is_absolute():
            alternatives = (
                MODULE_DIR / candidate,
                source_manifest_path.parent / candidate,
            )
            candidate = next((item for item in alternatives if item.exists()), alternatives[0])
        source_names[figure_name].add(candidate.name)
        if figure_name == "resonance_sweep_spectra" and candidate.suffix.lower() == ".npz":
            spectral_selections.add(row.get("selection", ""))
        if _source_file_valid(audit, candidate):
            source_count += 1
            if figure_name in ACTIVE_FIGURE_STEMS:
                traced_stems.add(figure_name)
            try:
                expected_size = int(row.get("source_size_bytes", ""))
            except ValueError:
                expected_size = -1
            audit.check(candidate.stat().st_size == expected_size,
                        "figure source size metadata",
                        "{} observed {}, declared {}".format(
                            candidate, candidate.stat().st_size, expected_size))
            observed_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
            audit.check(observed_hash == row.get("source_sha256", "").lower(),
                        "figure source SHA-256 metadata", str(candidate))
        for column, suffix in (("output_png", ".png"), ("output_pdf", ".pdf")):
            output = Path(row.get(column, ""))
            if not output.is_absolute():
                output = MODULE_DIR / output
            expected_output = FIGURE_DIR / (figure_name + suffix)
            audit.check(output.resolve() == expected_output.resolve() and output.is_file(),
                        "figure output manifest path", "{} -> {}".format(column, output))

    audit.check(source_names["wasserstein_line_scan"] == {
        "wasserstein_line_scan_source.csv",
        "local_baselines.csv",
        "wasserstein_peak_manifest.csv",
    }, "Wasserstein figure source inventory",
                "observed {}".format(sorted(source_names["wasserstein_line_scan"])))
    expected_sweep_sources = {
        "wasserstein_line_scan_source.csv",
        "resonance_sweep_spectra_manifest.csv",
    } | {case_id + ".npz" for case_id in sweep_ids}
    audit.check(source_names["resonance_sweep_spectra"] == expected_sweep_sources,
                "resonance-sweep figure source inventory",
                "observed {}, expected {}".format(
                    sorted(source_names["resonance_sweep_spectra"]),
                    sorted(expected_sweep_sources)))
    audit.check(spectral_selections == sweep_ids,
                "resonance-sweep figure archive selections",
                "manifest {}, table {}".format(
                    sorted(spectral_selections), sorted(sweep_ids)))

    baseline_source_rows = [
        row for row in source_rows
        if row.get("figure_stem", "") == "wasserstein_line_scan"
        and Path(row.get("source_file", "")).name == "local_baselines.csv"
    ]
    audit.check(len(baseline_source_rows) == 1,
                "Wasserstein baseline has one source-manifest row")
    if baseline_source_rows:
        baseline_row = baseline_source_rows[0]
        audit.check(
            baseline_row.get("selection", "")
            == ("primary J=0 rows; horizontal references for S=0.02,0.05,0.10 "
                "in both J-sign panels; all seven S values in inset"),
            "Wasserstein baseline selected-S/panel traceability",
        )
        audit.check(
            baseline_row.get("transformation", "")
            == "horizontal same-color lines for selected S; logarithmic inset for all S",
            "Wasserstein same-color horizontal-baseline traceability",
        )

    dense_sweep_rows = [
        row for row in source_rows
        if row.get("figure_stem", "") == "resonance_sweep_spectra"
        and Path(row.get("source_file", "")).name
        == "wasserstein_line_scan_source.csv"
    ]
    audit.check(
        len(dense_sweep_rows) == 1
        and dense_sweep_rows[0].get("selection", "")
        == "primary_fixed_S; J>0; S=0.05,0.10; 0.60<=r<=1.50 measured mesh"
        and dense_sweep_rows[0].get("transformation", "")
        == "measured points joined in r order; no interpolation or smoothing",
        "resonance-sweep dense measured-trace provenance",
    )
    audit.check(traced_stems == ACTIVE_FIGURE_STEMS,
                "all active command-16 figures trace to source data",
                "traced {}, expected {}".format(
                    sorted(traced_stems), sorted(ACTIVE_FIGURE_STEMS)))
    return {
        "figure_stems": sorted(png_stems & pdf_stems),
        "active_figure_stems": sorted(ACTIVE_FIGURE_STEMS),
        "command15_figure_stems": sorted(COMMAND15_FIGURE_STEMS),
        "command14_figure_stems": sorted(COMMAND14_FIGURE_STEMS),
        "source_count": source_count,
    }


def _trend_lines(summary: Sequence[Mapping[str, str]]) -> List[str]:
    lines: List[str] = []
    baselines = sorted(
        (row for row in summary if row.get("family") == "primary_J0_baseline"),
        key=lambda row: _float(row, "huang_rhys"),
    )
    if baselines:
        lines.append("J0 baseline L1 by S: " + ", ".join(
            "{:.2g}:{:.6f}".format(_float(row, "huang_rhys"), _float(row, "scba_l1"))
            for row in baselines
        ))
        passing_s = [_float(row, "huang_rhys") for row in baselines
                     if _truth(row.get("operational_validity_pass"))]
        lines.append("J0 operationally passing S values: {}".format(
            ", ".join("{:.2g}".format(value) for value in passing_s) if passing_s else "none"
        ))

    primary = [row for row in summary if row.get("family") == "primary_fixed_S"]
    for s_value in PRIMARY_S:
        for sign in (-1, 1):
            subset = [row for row in primary if _close(_float(row, "huang_rhys"), s_value)
                      and _int(row, "j_sign") == sign]
            if subset:
                passing = sum(_truth(row.get("operational_validity_pass")) for row in subset)
                l1_values = [_float(row, "scba_l1") for row in subset]
                lines.append(
                    "S={:.2g}, J{}: passing {}/{}, L1 range {:.6f}..{:.6f}".format(
                        s_value, "+" if sign > 0 else "-", passing, len(subset),
                        min(l1_values), max(l1_values),
                    )
                )

    weak_plus = sorted(
        (row for row in primary if _close(_float(row, "huang_rhys"), 0.02)
         and _int(row, "j_sign") == 1),
        key=lambda row: _float(row, "r_abs"),
    )
    if weak_plus:
        for center in (1.0, 3.0, 5.0):
            window = [row for row in weak_plus if abs(_float(row, "r_abs") - center) <= 0.20 + 1.0e-12]
            failed = [_float(row, "r_abs") for row in window
                      if not _truth(row.get("operational_validity_pass"))]
            maximum = max(window, key=lambda row: _float(row, "scba_l1")) if window else None
            if maximum is not None:
                lines.append(
                    "S=0.02,J+, bare odd guide r={:.0f}: max L1 {:.6f} at r={:.2f}; "
                    "composite failures {}".format(
                        center, _float(maximum, "scba_l1"), _float(maximum, "r_abs"),
                        ("{:.2f}..{:.2f}".format(min(failed), max(failed)) if failed else "none"),
                    )
                )

    fixed_lambda = [row for row in summary if row.get("family") == "secondary_fixed_lambda"]
    if fixed_lambda:
        for sign in (-1, 1):
            subset = [row for row in fixed_lambda if _int(row, "j_sign") == sign]
            lines.append(
                "fixed lambda=30 meV, J{}: passing {}/{}, L1 range {:.6f}..{:.6f}".format(
                    "+" if sign > 0 else "-",
                    sum(_truth(row.get("operational_validity_pass")) for row in subset),
                    len(subset), min(_float(row, "scba_l1") for row in subset),
                    max(_float(row, "scba_l1") for row in subset),
                )
            )
    return lines


def _build_report(
    audit: Audit,
    manifest: Sequence[Mapping[str, str]],
    summary: Sequence[Mapping[str, str]],
    cutoffs: Mapping[str, int],
    parity: Mapping[str, Any],
    validation_inventory: Mapping[str, Sequence[Mapping[str, str]]],
    metrics: Mapping[str, Any],
    archives: Mapping[str, Any],
    sweep_table_rows: int,
    representative_table_rows: int,
    line_scan_source_rows: int,
    legacy_representative_rows: int,
    legacy_frequency_map_rows: int,
    peaks: Mapping[str, Any],
    figures: Mapping[str, Any],
) -> str:
    status = "PASS" if not audit.failures else "FAIL"
    family_counts = Counter(row.get("family", "") for row in manifest)
    peak_status_counts = peaks.get("status_counts", {})
    peak_status_text = ", ".join(
        "J{} {}:{}".format("+" if sign > 0 else "-", extremum, count)
        for (sign, extremum), count in sorted(peak_status_counts.items())
    )
    lines = [
        "frequency-scan final audit",
        "==========================",
        "status = {}".format(status),
        "checks = {}".format(audit.checks),
        "failures = {}".format(len(audit.failures)),
        "",
        "Inventory",
        "---------",
        "manifest cases = {} (expected {})".format(len(manifest), EXPECTED_TOTAL),
        "summary cases = {} (expected {})".format(len(summary), EXPECTED_TOTAL),
        "family counts = {}".format(
            ", ".join("{}:{}".format(key, family_counts.get(key, 0))
                      for key in sorted(EXPECTED_FAMILIES))
        ),
        "primary fixed-S r points per S/sign = {}".format(EXPECTED_PRIMARY_R_POINTS),
        "sideband rows = {}".format(metrics.get("sideband_rows", 0)),
        "explicit no-candidate tracked-SCBA rows = {}".format(
            metrics.get("undefined_tracking_count", 0)),
        "local J0 baseline rows = 33 (7 fixed-S + 26 fixed-lambda)",
        "adaptive ED-cutoff rows/passing endpoints = {}/{}".format(
            cutoffs.get("rows", 0), cutoffs.get("passing_rows", 0)),
        "requested spectrum archives = {}".format(archives.get("archive_count", 0)),
        "resonance-sweep manifest rows = {}".format(sweep_table_rows),
        "preserved command-15 representative-manifest rows = {}".format(
            representative_table_rows),
        "Wasserstein line-scan source rows = {}".format(line_scan_source_rows),
        "Wasserstein peak-manifest rows = {}".format(peaks.get("rows", 0)),
        "Wasserstein peak statuses = {}".format(peak_status_text or "none"),
        "preserved legacy representative/frequency-map rows = {}/{}".format(
            legacy_representative_rows, legacy_frequency_map_rows),
        "exact level/crossing/validation rows = {}/{}/{}".format(
            parity.get("level_rows", 0), parity.get("crossing_rows", 0),
            parity.get("validation_rows", 0),
        ),
        "active command-16 figure stems = {}".format(
            ", ".join(figures.get("active_figure_stems", [])) or "none"),
        "preserved command-15 figure stems = {}".format(
            ", ".join(figures.get("command15_figure_stems", [])) or "none"),
        "preserved command-14 figure stems = {}".format(
            ", ".join(figures.get("command14_figure_stems", [])) or "none"),
        "figure source files checked = {}".format(figures.get("source_count", 0)),
        "",
        "Calculations and convergence audited",
        "-----------------------------------",
        "legacy/dimensionless/J0-scaling rows = {}/{}/{}".format(
            len(validation_inventory.get("legacy_reproduction.csv", [])),
            len(validation_inventory.get("dimensionless_scaling.csv", [])),
            len(validation_inventory.get("j0_scaling.csv", [])),
        ),
        "grid-padding/Poisson/parity rows = {}/{}/{}".format(
            len(validation_inventory.get("grid_padding_convergence.csv", [])),
            len(validation_inventory.get("poisson_validation.csv", [])),
            len(validation_inventory.get("parity_validation.csv", [])),
        ),
        "All generated checks and independently recomputed CSV tolerances must pass.",
        "All {} production rows must satisfy sum-rule, residual, causality, positivity, and channel-closure limits.".format(
            len(summary)),
        "",
        "Numerical extrema",
        "-----------------",
        "max raw ED/SCBA area relative mismatch = {:.9e}".format(metrics.get("raw_max", float("nan"))),
        "max ED pole-weight error = {:.9e}".format(metrics.get("pole_max", float("nan"))),
        "max common-mode omitted weight = {:.9e}".format(metrics.get("common_max", float("nan"))),
        "max ED first-moment error/omega0 = {:.9e}".format(metrics.get("moment_max", float("nan"))),
        "max SCBA fixed-point absolute residual/omega0 = {:.9e}".format(metrics.get("residual_abs_max", float("nan"))),
        "max SCBA fixed-point relative residual = {:.9e}".format(metrics.get("residual_rel_max", float("nan"))),
        "max positive Im Sigma/omega0 = {:.9e}".format(metrics.get("causality_max", float("nan"))),
        "minimum ED/SCBA spectra = {:.9e} / {:.9e}".format(
            metrics.get("ed_min", float("nan")), metrics.get("scba_min", float("nan"))),
        "max full-precision channel closure/omega0 = {:.9e}".format(metrics.get("closure_max", float("nan"))),
        "max archived complex64 channel closure absolute/relative = {:.9e} eV / {:.9e}".format(
            archives.get("archive_max_abs_closure_eV", float("nan")),
            archives.get("archive_max_rel_closure", float("nan")),
        ),
        "max forbidden exact parity-block coupling/omega0 = {:.9e}".format(
            parity.get("max_forbidden_coupling", float("nan"))),
        "",
        "Principal trends (data-derived, resolution-specific)",
        "----------------------------------------------------",
    ]
    lines.extend(_trend_lines(summary))
    lines.extend([
        "",
        "Scope and interpretation",
        "------------------------",
        "The operational gate is specific to eta/omega0=0.01 and is not a universal phase boundary.",
        "Rabi parity constrains exact difference-mode avoided crossings; it does not force the global SCBA error to be small at even controls.",
        "The fixed-S scan keeps the local dimensionless Franck-Condon problem fixed, whereas fixed lambda=30 meV follows S=0.6*r_abs and mixes mechanisms.",
        "Projected finite-eta self-energies are spectral diagnostics, not irreversible rates.",
    ])
    if audit.notes:
        lines.extend(["", "Notes", "-----"] + audit.notes)
    if audit.failures:
        lines.extend(["", "Failures", "--------"])
        lines.extend("- " + failure for failure in audit.failures[:200])
        if len(audit.failures) > 200:
            lines.append("- ... {} additional failures omitted".format(len(audit.failures) - 200))
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run every audit check without writing validation/frequency_scan/final_audit.txt",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    audit = Audit()

    manifest, summary, summary_by_id = _audit_manifest_and_summary(audit)
    cutoffs = _audit_cutoffs(audit, manifest)
    parity = _audit_exact_parity(audit)
    accuracy_gate, validation_inventory = _audit_generated_validation(audit)
    metrics = _audit_sidebands_and_metrics(audit, summary, summary_by_id, accuracy_gate)
    archives = _audit_archives(audit, manifest, summary_by_id)
    line_scan_source_rows = _audit_wasserstein_line_scan_source(audit, summary_by_id)
    legacy_frequency_map_rows = _audit_legacy_frequency_map_source(audit, summary_by_id)
    peaks = _audit_peak_manifest(audit, summary_by_id)
    representative_table_rows, _representative_ids = _audit_representative_table(
        audit, manifest, summary_by_id, peaks)
    sweep_table_rows, sweep_ids = _audit_resonance_sweep_table(
        audit, manifest, summary_by_id, peaks)
    legacy_representative_rows = _audit_legacy_representative_table(
        audit, manifest, summary_by_id)
    figures = _audit_figures(audit, sweep_ids)
    report = _build_report(
        audit, manifest, summary, cutoffs, parity, validation_inventory,
        metrics, archives, sweep_table_rows, representative_table_rows,
        line_scan_source_rows, legacy_representative_rows,
        legacy_frequency_map_rows, peaks, figures,
    )

    if not args.check_only:
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        temporary = REPORT_PATH.with_suffix(REPORT_PATH.suffix + ".tmp")
        temporary.write_text(report, encoding="utf-8")
        temporary.replace(REPORT_PATH)
    sys.stdout.write(report)
    return 0 if not audit.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
