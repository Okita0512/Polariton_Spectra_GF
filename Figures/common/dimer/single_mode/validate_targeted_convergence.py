#!/usr/bin/env python3
"""Run the three targeted Fig. S7--S8 cutoff/grid convergence checks.

The publication profile uses ``Delta x=0.002`` for
``x=(E-E_B)/nu0``.  This audit increases the factorized ED cutoff from 10
to 12 states and halves the common ED/SCBA grid spacing to 0.001.  It writes
validation metadata only and never overwrites production spectra.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import run_frequency_scan as scan  # noqa: E402
from calculate_single_mode_ed import solve_ed  # noqa: E402
from calculate_single_mode_scba import ScbaNumerics, solve_scba  # noqa: E402
from single_mode_common import ModelParams  # noqa: E402
from single_mode_metrics import compare_spectra  # noqa: E402


OUTPUT = HERE / "data" / "eta0p005"
CSV_PATH = OUTPUT / "targeted_convergence.csv"
JSON_PATH = OUTPUT / "targeted_convergence.json"
TEXT_PATH = OUTPUT / "targeted_convergence.txt"
MESH_PATH = OUTPUT / "mesh.csv"
BASELINE_PATH = OUTPUT / "baselines.csv"

DX_PRODUCTION = 0.002
DX_HALF = 0.001
ANALYSIS_X = (-18.0, 18.0)
INTERNAL_X = (-32.0, 24.0)
N_VIB_PRODUCTION = 10
N_VIB_INCREASED = 12

CASES = (
    ("J0_S0p10", "J=0 baseline", 0.10, 0.0, 50.0, 0.0),
    (
        "Jp_max_S0p10_r0p800", "measured positive-J maximum",
        0.10, 25.0, 62.5, 0.8,
    ),
    (
        "Jp_off_S0p10_r1p350", "positive-J off-integer reference",
        0.10, 25.0, 50.0 / 1.35, 1.35,
    ),
)

COLUMNS = (
    "case_id", "role", "huang_rhys", "j_meV", "r_abs", "nu0_meV",
    "eta_meV", "eta_over_nu0", "analysis_x_min", "analysis_x_max",
    "internal_x_min", "internal_x_max", "production_dx", "half_step_dx",
    "production_grid_samples", "half_step_grid_samples",
    "production_analysis_samples", "half_step_analysis_samples",
    "production_n_vib", "increased_n_vib", "production_matrix_dimension",
    "increased_matrix_dimension", "stored_w", "recalculated_production_w",
    "stored_w_absolute_difference", "increased_cutoff_w",
    "cutoff_delta_w", "half_step_w", "half_step_delta_w",
    "combined_increased_cutoff_half_step_w", "combined_delta_w",
    "ed_cutoff_unit_area_l1", "ed_cutoff_w_over_nu0",
    "ed_grid_unit_area_l1", "ed_grid_w_over_nu0",
    "scba_grid_unit_area_l1", "scba_grid_w_over_nu0",
    "production_ed_raw_area", "production_scba_raw_area",
    "half_step_ed_raw_area", "half_step_scba_raw_area",
    "production_scba_iterations", "half_step_scba_iterations",
    "production_scba_abs_residual_eV", "half_step_scba_abs_residual_eV",
    "production_scba_relative_residual", "half_step_scba_relative_residual",
    "production_scba_max_positive_im_sigma_eV",
    "half_step_scba_max_positive_im_sigma_eV", "runtime_s",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=str(path.parent),
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as handle:
        temporary = handle.name
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in COLUMNS})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_text(path: Path, value: str) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent),
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as handle:
        temporary = handle.name
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _refresh_profile_manifest() -> None:
    """Refresh hashes after replacing the targeted validation products."""

    import generate_eta0p005 as generator

    cuts = generator._read_csv(generator.CUT_PATH)
    generator._write_csv(
        generator.MANIFEST_PATH,
        generator._manifest(cuts),
        generator.MANIFEST_COLUMNS,
    )


def _stored_values() -> Dict[str, float]:
    mesh = _read_csv(MESH_PATH)
    baselines = _read_csv(BASELINE_PATH)
    return {
        "J0_S0p10": float(next(
            row["w0_W1_over_nu0"] for row in baselines
            if abs(float(row["huang_rhys"]) - 0.10) < 1.0e-12
        )),
        "Jp_max_S0p10_r0p800": float(next(
            row["w_W1_over_nu0"] for row in mesh
            if row["case_id"] == "eta005_S0p1_Jp_r0p8"
        )),
        "Jp_off_S0p10_r1p350": float(next(
            row["w_W1_over_nu0"] for row in mesh
            if row["case_id"] == "eta005_S0p1_Jp_r1p35"
        )),
    }


def _run_case(
    case_id: str,
    role: str,
    huang_rhys: float,
    j_meV: float,
    nu0_meV: float,
    r_abs: float,
    stored_w: float,
) -> Dict[str, Any]:
    started = time.perf_counter()
    model = ModelParams(
        energy_00_eV=2.0,
        omega0_meV=nu0_meV,
        huang_rhys=huang_rhys,
        j_meV=j_meV,
        temperature_K=0.0,
        eta_meV=0.005 * nu0_meV,
    )
    grids = {
        "production": scan.dimensionless_grid(model, dx=DX_PRODUCTION),
        "half_step": scan.dimensionless_grid(model, dx=DX_HALF),
    }
    solutions: Dict[Tuple[Any, ...], Any] = {}
    controls = ScbaNumerics()
    for grid_name, grid in grids.items():
        for n_vib in (N_VIB_PRODUCTION, N_VIB_INCREASED):
            solutions[("ed", grid_name, n_vib)] = solve_ed(
                model, grid, n_vib=n_vib, n_thermal=1,
                thermal_tolerance=1.0e-12,
                stick_weight_tolerance=1.0e-14,
                deposition_refinement=4,
            )
        solutions[("scba", grid_name)] = solve_scba(
            model, grid, controls, seed=None,
            use_zero_temperature_sweep=True, verbose=False,
        )

    def closure(grid_name: str, n_vib: int) -> Any:
        grid = grids[grid_name]
        ed = solutions[("ed", grid_name, n_vib)]
        scba = solutions[("scba", grid_name)]
        return compare_spectra(
            ed.energy_eV, ed.spectrum_bright,
            scba.spectrum_bright_per_eV,
            window=(grid.analysis_min_eV, grid.analysis_max_eV),
        )

    production = closure("production", N_VIB_PRODUCTION)
    cutoff = closure("production", N_VIB_INCREASED)
    half_step = closure("half_step", N_VIB_PRODUCTION)
    combined = closure("half_step", N_VIB_INCREASED)
    production_ed = solutions[("ed", "production", N_VIB_PRODUCTION)]
    production_scba = solutions[("scba", "production")]
    fine_ed = solutions[("ed", "half_step", N_VIB_PRODUCTION)]
    fine_scba = solutions[("scba", "half_step")]
    energy = production_ed.energy_eV
    window = (
        grids["production"].analysis_min_eV,
        grids["production"].analysis_max_eV,
    )
    ed_cutoff_change = compare_spectra(
        energy, production_ed.spectrum_bright,
        solutions[("ed", "production", N_VIB_INCREASED)].spectrum_bright,
        window=window,
    )
    ed_grid_change = compare_spectra(
        energy, production_ed.spectrum_bright,
        np.interp(energy, fine_ed.energy_eV, fine_ed.spectrum_bright),
        window=window,
    )
    scba_grid_change = compare_spectra(
        energy, production_scba.spectrum_bright_per_eV,
        np.interp(
            energy, fine_scba.omega_eV,
            fine_scba.spectrum_bright_per_eV,
        ),
        window=window,
    )
    omega0_eV = model.omega0_eV
    production_w = production.wasserstein_eV / omega0_eV
    cutoff_w = cutoff.wasserstein_eV / omega0_eV
    half_step_w = half_step.wasserstein_eV / omega0_eV
    combined_w = combined.wasserstein_eV / omega0_eV
    return {
        "case_id": case_id, "role": role, "huang_rhys": huang_rhys,
        "j_meV": j_meV, "r_abs": r_abs, "nu0_meV": nu0_meV,
        "eta_meV": model.eta_meV, "eta_over_nu0": model.eta_eV / omega0_eV,
        "analysis_x_min": ANALYSIS_X[0], "analysis_x_max": ANALYSIS_X[1],
        "internal_x_min": INTERNAL_X[0], "internal_x_max": INTERNAL_X[1],
        "production_dx": DX_PRODUCTION, "half_step_dx": DX_HALF,
        "production_grid_samples": production_ed.energy_eV.size,
        "half_step_grid_samples": fine_ed.energy_eV.size,
        "production_analysis_samples": int(np.count_nonzero(
            grids["production"].analysis_mask(production_ed.energy_eV)
        )),
        "half_step_analysis_samples": int(np.count_nonzero(
            grids["half_step"].analysis_mask(fine_ed.energy_eV)
        )),
        "production_n_vib": N_VIB_PRODUCTION,
        "increased_n_vib": N_VIB_INCREASED,
        "production_matrix_dimension": 2 * N_VIB_PRODUCTION,
        "increased_matrix_dimension": 2 * N_VIB_INCREASED,
        "stored_w": stored_w, "recalculated_production_w": production_w,
        "stored_w_absolute_difference": abs(production_w - stored_w),
        "increased_cutoff_w": cutoff_w,
        "cutoff_delta_w": cutoff_w - production_w,
        "half_step_w": half_step_w,
        "half_step_delta_w": half_step_w - production_w,
        "combined_increased_cutoff_half_step_w": combined_w,
        "combined_delta_w": combined_w - production_w,
        "ed_cutoff_unit_area_l1": ed_cutoff_change.unit_area_l1,
        "ed_cutoff_w_over_nu0": ed_cutoff_change.wasserstein_eV / omega0_eV,
        "ed_grid_unit_area_l1": ed_grid_change.unit_area_l1,
        "ed_grid_w_over_nu0": ed_grid_change.wasserstein_eV / omega0_eV,
        "scba_grid_unit_area_l1": scba_grid_change.unit_area_l1,
        "scba_grid_w_over_nu0": scba_grid_change.wasserstein_eV / omega0_eV,
        "production_ed_raw_area": production.reference_raw_area,
        "production_scba_raw_area": production.candidate_raw_area,
        "half_step_ed_raw_area": half_step.reference_raw_area,
        "half_step_scba_raw_area": half_step.candidate_raw_area,
        "production_scba_iterations": production_scba.iterations,
        "half_step_scba_iterations": fine_scba.iterations,
        "production_scba_abs_residual_eV":
            production_scba.fixed_point_abs_residual_eV,
        "half_step_scba_abs_residual_eV": fine_scba.fixed_point_abs_residual_eV,
        "production_scba_relative_residual":
            production_scba.fixed_point_relative_residual,
        "half_step_scba_relative_residual": fine_scba.fixed_point_relative_residual,
        "production_scba_max_positive_im_sigma_eV":
            production_scba.max_positive_im_sigma_eV,
        "half_step_scba_max_positive_im_sigma_eV":
            fine_scba.max_positive_im_sigma_eV,
        "runtime_s": time.perf_counter() - started,
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stored = _stored_values()
    rows = [_run_case(*case, stored[case[0]]) for case in CASES]
    delta_keys = ("cutoff_delta_w", "half_step_delta_w", "combined_delta_w")
    maximum_numerical_delta = max(
        abs(float(row[key])) for row in rows for key in delta_keys
    )
    maximum_ratio = max(
        max(abs(float(row[key])) for key in delta_keys) / float(row["stored_w"])
        for row in rows
    )
    maxima = {
        "stored_w_absolute_difference": max(
            float(row["stored_w_absolute_difference"]) for row in rows
        ),
        "absolute_cutoff_delta_w": max(
            abs(float(row["cutoff_delta_w"])) for row in rows
        ),
        "absolute_half_step_delta_w": max(
            abs(float(row["half_step_delta_w"])) for row in rows
        ),
        "absolute_combined_delta_w": max(
            abs(float(row["combined_delta_w"])) for row in rows
        ),
        "numerical_delta_w_over_displayed_w": maximum_ratio,
        "ed_cutoff_unit_area_l1": max(
            float(row["ed_cutoff_unit_area_l1"]) for row in rows
        ),
        "ed_grid_unit_area_l1": max(
            float(row["ed_grid_unit_area_l1"]) for row in rows
        ),
        "scba_grid_unit_area_l1": max(
            float(row["scba_grid_unit_area_l1"]) for row in rows
        ),
        "ed_cutoff_w_over_nu0": max(
            float(row["ed_cutoff_w_over_nu0"]) for row in rows
        ),
        "ed_grid_w_over_nu0": max(
            float(row["ed_grid_w_over_nu0"]) for row in rows
        ),
        "scba_grid_w_over_nu0": max(
            float(row["scba_grid_w_over_nu0"]) for row in rows
        ),
        "scba_abs_residual_eV": max(
            float(row[key]) for row in rows for key in (
                "production_scba_abs_residual_eV",
                "half_step_scba_abs_residual_eV",
            )
        ),
        "scba_relative_residual": max(
            float(row[key]) for row in rows for key in (
                "production_scba_relative_residual",
                "half_step_scba_relative_residual",
            )
        ),
        "positive_im_sigma_eV": max(
            float(row[key]) for row in rows for key in (
                "production_scba_max_positive_im_sigma_eV",
                "half_step_scba_max_positive_im_sigma_eV",
            )
        ),
    }
    checks = [
        ("production_reproduces_stored_w", maxima["stored_w_absolute_difference"], 5.0e-10),
        ("numerical_delta_below_one_percent_of_displayed_error", maximum_ratio, 1.0e-2),
        ("scba_absolute_fixed_point_residual", maxima["scba_abs_residual_eV"], 1.0e-9),
        ("scba_relative_fixed_point_residual", maxima["scba_relative_residual"], 1.0e-8),
        ("retarded_self_energy_causality", maxima["positive_im_sigma_eV"], 5.0e-10),
    ]
    check_rows = [
        {"name": name, "value": value, "threshold": threshold,
         "pass": value <= threshold}
        for name, value, threshold in checks
    ]
    payload = {
        "schema_version": "figs6_s7_targeted_convergence_v1",
        "generated_utc": _utc_now(),
        "command": "python3 Figures/common/dimer/single_mode/validate_targeted_convergence.py",
        "profile": {
            "analysis_x": ANALYSIS_X, "internal_x": INTERNAL_X,
            "eta_over_nu0": 0.005, "production_dx": DX_PRODUCTION,
            "half_step_dx": DX_HALF,
            "production_n_vib": N_VIB_PRODUCTION,
            "increased_n_vib": N_VIB_INCREASED,
        },
        "case_count": len(rows), "maxima": maxima,
        "maximum_numerical_delta_w": maximum_numerical_delta,
        "checks": check_rows,
        "pass": all(bool(item["pass"]) for item in check_rows),
    }
    _write_csv(CSV_PATH, rows)
    _write_text(JSON_PATH, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "Fig. S7-S8 targeted cutoff/grid convergence",
        "generated_utc=%s" % payload["generated_utc"],
        "command=%s" % payload["command"],
        "production Delta x=0.002; half-step Delta x=0.001",
        "ED N_vib=10 -> 12; difference-mode matrix dimension 20 -> 24",
    ]
    for row in rows:
        lines.append(
            "%s stored_w=%.12g cutoff_delta_w=%+.12e "
            "half_step_delta_w=%+.12e combined_delta_w=%+.12e" % (
                row["case_id"], row["stored_w"], row["cutoff_delta_w"],
                row["half_step_delta_w"], row["combined_delta_w"],
            )
        )
    for item in check_rows:
        lines.append(
            "%s %s value=%.12e threshold<=%.12e" % (
                "PASS" if item["pass"] else "FAIL", item["name"],
                item["value"], item["threshold"],
            )
        )
    lines.extend([
        "maximum_numerical_delta_w=%.12e" % maximum_numerical_delta,
        "maximum_delta_over_displayed_w=%.12e" % maximum_ratio,
        "status = %s" % ("PASS" if payload["pass"] else "FAIL"),
        "failures = %d" % sum(not bool(item["pass"]) for item in check_rows),
    ])
    _write_text(TEXT_PATH, "\n".join(lines) + "\n")
    _refresh_profile_manifest()
    print("targeted convergence: %s" % ("PASS" if payload["pass"] else "FAIL"))
    print("maximum |Delta w| = %.12e" % maximum_numerical_delta)
    print("maximum |Delta w| / displayed w = %.12e" % maximum_ratio)
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
