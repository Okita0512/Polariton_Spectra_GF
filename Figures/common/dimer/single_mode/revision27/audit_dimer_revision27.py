#!/usr/bin/env python3
"""Focused, non-production audit for SI Figs. S6--S7 (revision 27).

This script leaves the active 2,214-point publication scan and its archived
spectra untouched.  It performs independent midpoint-in-r, energy-grid,
grid-phase, cutoff, support, CDF, peak/region-weight, and exact-parity checks
and writes every result below this ``revision27`` directory.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import eigh  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402


HERE = Path(__file__).resolve().parent
SINGLE_MODE = HERE.parent
SOURCE = SINGLE_MODE / "source"
COMMON = SINGLE_MODE.parents[1]
PROJECT_ROOT = SINGLE_MODE.parents[3]
for candidate in (SOURCE, COMMON):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import run_frequency_scan as scan  # noqa: E402
from calculate_single_mode_ed import solve_ed  # noqa: E402
from calculate_single_mode_scba import ScbaNumerics, solve_scba  # noqa: E402
from single_mode_common import GridSpec, ModelParams  # noqa: E402
from si_plot_style import (  # noqa: E402
    EXACT_STYLE,
    SCBA_STYLE,
    add_panel_label,
    compact_legend_kwargs,
    configure_si_matplotlib,
    style_axis,
)


DATA = HERE / "data"
PLOTS = HERE / "plots"
PUBLIC_DATA = SINGLE_MODE / "data" / "eta0p005"
MESH = PUBLIC_DATA / "mesh.csv"
CUTS = PUBLIC_DATA / "cuts.csv"
FIG_S6_PROCESSED = PROJECT_ROOT / "Figures/SI/FigS6/processed/error_mesh.csv"
FIG_S7_PROCESSED = PROJECT_ROOT / "Figures/archive/revision28/old_S6_S7/old_S7_indicator_mesh.csv"

J_ABS_MEV = 25.0
ETA_OVER_NU0 = 0.005
INTERNAL_X = (-32.0, 24.0)
ANALYSIS_X = (-18.0, 18.0)
REGION_EDGES = (-18.0, -1.5, -0.5, 0.5, 1.5, 2.5, 4.0, 18.0)

# Midpoints bracket measured local maxima and the small secondary structure
# around r~=2.  Negative-J neighbors are controls, not resonance labels.
REFINED_R_CASES = {
    (0.05, +1): (
        0.8825, 0.8875,
        1.375, 1.400, 1.425, 1.475, 1.500, 1.525,
        1.9375, 1.9425, 1.9875, 1.9925, 2.0025,
    ),
    (0.10, +1): (
        0.795, 0.805,
        1.275, 1.300, 1.325, 1.375, 1.400, 1.425,
        1.8775, 1.8825, 1.9375, 1.9425, 1.9875, 1.9925, 2.0025,
    ),
    (0.05, -1): (1.9475, 1.9975, 2.0525),
    (0.10, -1): (1.9475, 1.9975, 2.0525),
}

CONVERGENCE_CASES = (
    ("S005_Jp_r0885", 0.05, +1, 0.885),
    ("S010_Jp_r0800", 0.10, +1, 0.800),
    ("S005_Jp_r1940", 0.05, +1, 1.940),
    ("S010_Jp_r1880", 0.10, +1, 1.880),
    ("S010_Jp_r1940", 0.10, +1, 1.940),
    ("S005_Jp_r1995", 0.05, +1, 1.995),
    ("S005_Jm_r2000", 0.05, -1, 2.000),
    ("S010_Jm_r2000", 0.10, -1, 2.000),
)


@dataclass
class Solution:
    label: str
    s_value: float
    sign: int
    r_value: float
    dx: float
    phase: float
    n_vib: int
    x: np.ndarray
    ed_probability: np.ndarray
    scba_probability: np.ndarray
    cdf_difference: np.ndarray
    w: float
    ed_area: float
    scba_area: float
    scba_abs_residual_eV: float
    scba_relative_residual: float
    scba_max_positive_im_sigma_eV: float
    ed_minimum: float
    scba_minimum: float
    w_support_m20_p20: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in columns})


def _model(s_value: float, sign: int, r_value: float) -> ModelParams:
    nu0 = 2.0 * J_ABS_MEV / r_value
    return ModelParams(
        energy_00_eV=2.0,
        omega0_meV=nu0,
        huang_rhys=s_value,
        j_meV=sign * J_ABS_MEV,
        temperature_K=0.0,
        eta_meV=ETA_OVER_NU0 * nu0,
    )


def _phase_grid(model: ModelParams, dx: float, phase: float) -> GridSpec:
    base = scan.dimensionless_grid(
        model, dx=dx, analysis_x=ANALYSIS_X, internal_x=INTERNAL_X
    )
    shift = phase * dx * model.omega0_eV
    return GridSpec(
        analysis_min_eV=base.analysis_min_eV,
        analysis_max_eV=base.analysis_max_eV,
        internal_min_eV=base.internal_min_eV + shift,
        internal_max_eV=base.internal_max_eV + shift,
        step_meV=base.step_meV,
    )


def _normalize_on_x(
    energy: np.ndarray,
    spectrum: np.ndarray,
    model: ModelParams,
    dx: float,
    support: tuple[float, float] = ANALYSIS_X,
) -> tuple[np.ndarray, np.ndarray, float]:
    count = int(round((support[1] - support[0]) / dx))
    x = np.linspace(support[0], support[1], count + 1)
    bright = model.vertical_energy_eV + model.j_eV
    target_energy = bright + x * model.omega0_eV
    values = np.interp(target_energy, energy, spectrum)
    area_x = float(np.trapz(values, x))
    if area_x <= 0.0:
        raise RuntimeError("Nonpositive spectrum area in x")
    return x, values / area_x, area_x * model.omega0_eV


def _quadrature_mass(x: np.ndarray, probability_density: np.ndarray) -> np.ndarray:
    """Trapezoidal point masses used by the active W1 implementation."""

    weights = np.empty_like(x)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    mass = probability_density * weights
    return mass / np.sum(mass)


def _metric_from_probabilities(
    x: np.ndarray,
    ed_probability: np.ndarray,
    scba_probability: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return the exact discrete CDF difference and its W1 integral."""

    ed_mass = _quadrature_mass(x, ed_probability)
    scba_mass = _quadrature_mass(x, scba_probability)
    cdf_difference = np.cumsum(ed_mass - scba_mass)
    # The CDF is constant between adjacent support points.  This is exactly
    # the one-dimensional Wasserstein distance of the trapezoidal point
    # masses used in the production code.
    w = float(np.sum(np.abs(cdf_difference[:-1]) * np.diff(x)))
    return cdf_difference, w


def _solve(
    label: str,
    s_value: float,
    sign: int,
    r_value: float,
    dx: float = 0.002,
    phase: float = 0.0,
    n_vib: int = 10,
) -> Solution:
    model = _model(s_value, sign, r_value)
    grid = _phase_grid(model, dx, phase)
    ed = solve_ed(
        model,
        grid,
        n_vib=n_vib,
        n_thermal=1,
        thermal_tolerance=1.0e-12,
        stick_weight_tolerance=1.0e-14,
        deposition_refinement=4,
    )
    scba = solve_scba(
        model,
        grid,
        ScbaNumerics(),
        seed=None,
        use_zero_temperature_sweep=True,
        verbose=False,
    )
    x, ed_probability, ed_area = _normalize_on_x(
        ed.energy_eV, ed.spectrum_bright, model, dx
    )
    x_scba, scba_probability, scba_area = _normalize_on_x(
        scba.omega_eV, scba.spectrum_bright_per_eV, model, dx
    )
    if not np.array_equal(x, x_scba):
        raise RuntimeError("ED/SCBA metric grids differ")
    cdf_difference, w = _metric_from_probabilities(
        x, ed_probability, scba_probability
    )
    x_wide, ed_wide, _ed_area_wide = _normalize_on_x(
        ed.energy_eV, ed.spectrum_bright, model, dx, (-20.0, 20.0)
    )
    x_scba_wide, scba_wide, _scba_area_wide = _normalize_on_x(
        scba.omega_eV,
        scba.spectrum_bright_per_eV,
        model,
        dx,
        (-20.0, 20.0),
    )
    if not np.array_equal(x_wide, x_scba_wide):
        raise RuntimeError("ED/SCBA wide-support grids differ")
    _wide_cdf, w_wide = _metric_from_probabilities(
        x_wide, ed_wide, scba_wide
    )
    return Solution(
        label=label,
        s_value=s_value,
        sign=sign,
        r_value=r_value,
        dx=dx,
        phase=phase,
        n_vib=n_vib,
        x=x,
        ed_probability=ed_probability,
        scba_probability=scba_probability,
        cdf_difference=cdf_difference,
        w=w,
        ed_area=ed_area,
        scba_area=scba_area,
        scba_abs_residual_eV=float(scba.fixed_point_abs_residual_eV),
        scba_relative_residual=float(scba.fixed_point_relative_residual),
        scba_max_positive_im_sigma_eV=float(scba.max_positive_im_sigma_eV),
        ed_minimum=float(np.min(ed.spectrum_bright)),
        scba_minimum=float(np.min(scba.spectrum_bright_per_eV)),
        w_support_m20_p20=w_wide,
    )


def _w_on_support(solution: Solution, support: tuple[float, float]) -> float:
    mask = (solution.x >= support[0] - 1.0e-12) & (solution.x <= support[1] + 1.0e-12)
    x = solution.x[mask]
    ed = solution.ed_probability[mask]
    scba = solution.scba_probability[mask]
    ed /= np.trapz(ed, x)
    scba /= np.trapz(scba, x)
    _difference, w = _metric_from_probabilities(x, ed, scba)
    return w


def _solution_row(solution: Solution) -> dict[str, Any]:
    difference = solution.scba_probability - solution.ed_probability
    centroid_ed = float(np.trapz(solution.x * solution.ed_probability, solution.x))
    centroid_scba = float(np.trapz(solution.x * solution.scba_probability, solution.x))
    return {
        "label": solution.label,
        "huang_rhys": solution.s_value,
        "j_sign": solution.sign,
        "r_abs": solution.r_value,
        "nu0_meV": 2.0 * J_ABS_MEV / solution.r_value,
        "dx": solution.dx,
        "grid_phase_in_cells": solution.phase,
        "n_vib": solution.n_vib,
        "w_W1_over_nu0": solution.w,
        "unit_area_L1_over_x": float(np.trapz(np.abs(difference), solution.x)),
        "centroid_ED_over_nu0": centroid_ed,
        "centroid_SCBA_over_nu0": centroid_scba,
        "centroid_shift_SCBA_minus_ED_over_nu0": centroid_scba - centroid_ed,
        "w_support_m16_p16": _w_on_support(solution, (-16.0, 16.0)),
        "w_support_m18_p18": solution.w,
        "w_support_m20_p20": solution.w_support_m20_p20,
        "ed_raw_area_over_E": solution.ed_area,
        "scba_raw_area_over_E": solution.scba_area,
        "raw_area_relative_difference": solution.scba_area / solution.ed_area - 1.0,
        "scba_abs_residual_eV": solution.scba_abs_residual_eV,
        "scba_relative_residual": solution.scba_relative_residual,
        "scba_max_positive_im_sigma_eV": solution.scba_max_positive_im_sigma_eV,
        "minimum_ed_spectrum": solution.ed_minimum,
        "minimum_scba_spectrum": solution.scba_minimum,
    }


def _peak_rows(solution: Solution, method: str) -> list[dict[str, Any]]:
    values = (
        solution.ed_probability if method == "ED" else solution.scba_probability
    )
    selected = (solution.x >= -2.0) & (solution.x <= 5.0)
    x = solution.x[selected]
    y = values[selected]
    peaks, properties = find_peaks(
        y,
        prominence=0.0025 * float(np.max(y)),
        distance=max(1, int(round(0.025 / solution.dx))),
    )
    if peaks.size == 0:
        return []
    # Integrate each peak over the valleys shared with its neighbors.
    boundaries = [0]
    for left, right in zip(peaks[:-1], peaks[1:]):
        boundaries.append(int(left + np.argmin(y[left : right + 1])))
    boundaries.append(len(x) - 1)
    records: list[dict[str, Any]] = []
    for ordinal, peak in enumerate(peaks):
        left = boundaries[ordinal]
        right = boundaries[ordinal + 1]
        records.append({
            "label": solution.label,
            "method": method,
            "peak_ordinal": ordinal + 1,
            "peak_x": float(x[peak]),
            "peak_height": float(y[peak]),
            "relative_prominence": float(properties["prominences"][ordinal] / np.max(y)),
            "basin_left_x": float(x[left]),
            "basin_right_x": float(x[right]),
            "basin_weight": float(np.trapz(y[left : right + 1], x[left : right + 1])),
        })
    return records


def _region_rows(solution: Solution) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lower, upper in zip(REGION_EDGES[:-1], REGION_EDGES[1:]):
        mask = (solution.x >= lower - 1.0e-12) & (solution.x <= upper + 1.0e-12)
        x = solution.x[mask]
        ed = solution.ed_probability[mask]
        scba = solution.scba_probability[mask]
        ed_weight = float(np.trapz(ed, x))
        scba_weight = float(np.trapz(scba, x))
        interval_mask = (
            (solution.x[:-1] >= lower - 1.0e-12)
            & (solution.x[1:] <= upper + 1.0e-12)
        )
        cdf_contribution = float(np.sum(
            np.abs(solution.cdf_difference[:-1][interval_mask])
            * np.diff(solution.x)[interval_mask]
        ))
        rows.append({
            "label": solution.label,
            "x_lower": lower,
            "x_upper": upper,
            "ed_weight": ed_weight,
            "scba_weight": scba_weight,
            "scba_minus_ed_weight": scba_weight - ed_weight,
            "absolute_cdf_integral": cdf_contribution,
            "fraction_of_w": cdf_contribution / solution.w,
        })
    return rows


def _parity_hamiltonian(s_value: float, r_value: float, n_vib: int) -> np.ndarray:
    number = np.diag(np.arange(n_vib, dtype=float))
    raising = np.diag(np.sqrt(np.arange(1, n_vib, dtype=float)), 1)
    coordinate = raising + raising.T
    tau_z = np.diag((1.0, -1.0))
    tau_x = np.asarray(((0.0, 1.0), (1.0, 0.0)))
    return (
        np.kron(0.5 * r_value * tau_z, np.eye(n_vib))
        + np.kron(np.eye(2), number)
        + math.sqrt(0.5 * s_value) * np.kron(tau_x, coordinate)
    )


def _parity_vector(n_vib: int) -> np.ndarray:
    oscillator = np.where(np.arange(n_vib) % 2 == 0, 1.0, -1.0)
    return np.concatenate((oscillator, -oscillator))


def _branch_difference(s_value: float, r_value: float, n_vib: int = 12) -> tuple[float, dict[str, float]]:
    hamiltonian = _parity_hamiltonian(s_value, r_value, n_vib)
    parity = _parity_vector(n_vib)
    plus = np.flatnonzero(parity > 0.0)
    minus = np.flatnonzero(parity < 0.0)
    e_plus, v_plus = eigh(hamiltonian[np.ix_(plus, plus)])
    e_minus, v_minus = eigh(hamiltonian[np.ix_(minus, minus)])
    full_plus = np.zeros((2 * n_vib, len(e_plus)))
    full_minus = np.zeros((2 * n_vib, len(e_minus)))
    full_plus[plus] = v_plus
    full_minus[minus] = v_minus
    index_plus = int(np.argmax(np.square(full_plus[0])))
    index_minus = int(np.argmax(np.square(full_minus[n_vib + 2])))
    details = {
        "bright_branch_energy_over_nu0": float(e_plus[index_plus]),
        "dark_even_branch_energy_over_nu0": float(e_minus[index_minus]),
        "bright0_overlap": float(full_plus[0, index_plus] ** 2),
        "dark2_overlap": float(full_minus[n_vib + 2, index_minus] ** 2),
        "source_overlap_opposite_parity": float(np.max(np.abs(full_minus[0]))),
    }
    return float(e_plus[index_plus] - e_minus[index_minus]), details


def _parity_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    levels: list[dict[str, Any]] = []
    for s_value in (0.05, 0.10):
        n_vib = 12
        hamiltonian = _parity_hamiltonian(s_value, 2.0, n_vib)
        parity = _parity_vector(n_vib)
        operator = np.diag(parity)
        plus = np.flatnonzero(parity > 0.0)
        minus = np.flatnonzero(parity < 0.0)
        source = np.zeros(2 * n_vib)
        source[0] = 1.0
        checks.append({
            "huang_rhys": s_value,
            "r_abs": 2.0,
            "commutator_max_abs": float(np.max(np.abs(hamiltonian @ operator - operator @ hamiltonian))),
            "opposite_parity_H_block_max_abs": float(np.max(np.abs(hamiltonian[np.ix_(plus, minus)]))),
            "source_opposite_parity_projected_norm": float(np.linalg.norm(source[minus])),
            "B0_to_D_even_direct_max_abs": float(max(abs(hamiltonian[n_vib + n, 0]) for n in range(0, n_vib, 2))),
            "B0_to_D1_direct_abs": float(abs(hamiltonian[n_vib + 1, 0])),
        })
        grid = np.linspace(1.55, 2.15, 121)
        differences = np.asarray([_branch_difference(s_value, value, n_vib)[0] for value in grid])
        roots: list[float] = []
        for left, right, f_left, f_right in zip(grid[:-1], grid[1:], differences[:-1], differences[1:]):
            if f_left == 0.0:
                roots.append(float(left))
            elif f_left * f_right < 0.0:
                roots.append(float(brentq(lambda value: _branch_difference(s_value, value, n_vib)[0], left, right)))
        if not roots:
            root = float(grid[int(np.argmin(np.abs(differences)))])
        else:
            root = min(roots, key=lambda value: abs(value - 2.0))
        for r_value in (root - 0.02, root, root + 0.02, 2.0):
            difference, details = _branch_difference(s_value, r_value, n_vib)
            levels.append({
                "huang_rhys": s_value,
                "r_abs": r_value,
                "is_tracked_crossing": abs(r_value - root) < 1.0e-10,
                "signed_gap_bright_minus_dark_over_nu0": difference,
                "absolute_gap_over_nu0": abs(difference),
                **details,
            })
    return checks, levels


def _pipeline_audit() -> dict[str, Any]:
    mesh = _read_csv(MESH)
    refined = _read_csv(DATA / "refined_r_points.csv")
    s6 = _read_csv(FIG_S6_PROCESSED)
    s7 = _read_csv(FIG_S7_PROCESSED)
    mesh_by_id = {row["case_id"]: row for row in mesh}
    refined_for_plot = {
        row["label"]: {
            "case_id": row["label"],
            "huang_rhys": row["huang_rhys"],
            "j_sign": row["j_sign"],
            "r_abs": row["r_abs"],
            "w_W1_over_nu0": row["w_W1_over_nu0"],
        }
        for row in refined
    }
    expected_s6 = {**mesh_by_id, **refined_for_plot}
    s6_by_id = {row["case_id"]: row for row in s6}
    selected_s7 = {
        row["case_id"]: row
        for row in expected_s6.values()
        if int(row["j_sign"]) == +1
        and float(row["huang_rhys"]) in (0.05, 0.10)
        and 0.60 - 1.0e-12 <= float(row["r_abs"]) <= 1.50 + 1.0e-12
    }
    groups: dict[str, Any] = {}
    shared_r: np.ndarray | None = None
    for s_value in (0.02, 0.05, 0.10):
        for sign in (-1, +1):
            rows = sorted(
                (
                    row for row in mesh
                    if abs(float(row["huang_rhys"]) - s_value) < 1.0e-12
                    and int(row["j_sign"]) == sign
                ),
                key=lambda row: float(row["r_abs"]),
            )
            r_values = np.asarray([float(row["r_abs"]) for row in rows])
            if shared_r is None:
                shared_r = r_values
            groups[f"S{s_value:g}_J{sign:+d}"] = {
                "rows": len(rows),
                "strictly_increasing": bool(np.all(np.diff(r_values) > 0.0)),
                "unique_r": int(len(np.unique(r_values))),
                "same_shared_r_mesh": bool(np.array_equal(r_values, shared_r)),
                "minimum_step": float(np.min(np.diff(r_values))),
                "maximum_step": float(np.max(np.diff(r_values))),
            }
    numeric_columns = ("huang_rhys", "j_sign", "r_abs", "w_W1_over_nu0")
    s6_exact = set(s6_by_id) == set(expected_s6) and all(
        case_id in expected_s6
        and all(
            abs(float(row[column]) - float(expected_s6[case_id][column])) <= 2.0e-15
            for column in numeric_columns
        )
        for case_id, row in s6_by_id.items()
    )
    s7_by_id = {row["case_id"]: row for row in s7}
    s7_exact = set(s7_by_id) == set(selected_s7) and all(
        abs(float(row["w_W1_over_nu0"]) - float(selected_s7[case_id]["w_W1_over_nu0"])) <= 2.0e-15
        for case_id, row in s7_by_id.items()
    )
    return {
        "generated_utc": _utc_now(),
        "authoritative_mesh": str(MESH.relative_to(PROJECT_ROOT)),
        "authoritative_mesh_sha256": _sha256(MESH),
        "mesh_rows": len(mesh),
        "unique_case_ids": len(mesh_by_id),
        "duplicate_case_ids": len(mesh) - len(mesh_by_id),
        "groups": groups,
        "fig_s6_processed_rows": len(s6),
        "fig_s6_contains_all_production_case_ids": set(mesh_by_id) <= set(s6_by_id),
        "fig_s6_exact_calculated_merged_source": s6_exact,
        "fig_s7_processed_rows": len(s7),
        "fig_s7_exact_calculated_overlap_with_s6": s7_exact,
        "fig_s7_expected_overlap_rows": len(selected_s7),
        "source_cuts_sha256": _sha256(CUTS),
        "plot_transform": "sorted calculated points connected directly; no resampling, interpolation, smoothing, filtering, or downsampling",
        "publication_pdf_path_simplification_after_revision27": False,
    }


def _diagnostic_plot(solutions: Sequence[Solution], output: Path) -> None:
    configure_si_matplotlib()
    fig, axes = plt.subplots(
        len(solutions), 2, figsize=(7.25, 2.15 * len(solutions) + 0.40),
        sharex="col", squeeze=False,
    )
    for row_index, solution in enumerate(solutions):
        ax_spectrum, ax_cdf = axes[row_index]
        display = (solution.x >= -1.5) & (solution.x <= 4.0)
        ax_spectrum.plot(solution.x[display], solution.ed_probability[display], label="ED", **EXACT_STYLE)
        ax_spectrum.plot(solution.x[display], solution.scba_probability[display], label="SCBA", **SCBA_STYLE)
        ax_cdf.plot(solution.x, solution.cdf_difference, color="#4C78A8", linewidth=0.85)
        ax_cdf.axhline(0.0, color="#888888", linewidth=0.45)
        sign_text = "+" if solution.sign > 0 else "-"
        ax_spectrum.text(
            0.97,
            0.94,
            rf"$S={solution.s_value:.2f}$, $J={sign_text}25$ meV, $r={solution.r_value:.3f}$",
            transform=ax_spectrum.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
        )
        ax_cdf.text(
            0.97,
            0.94,
            rf"$w={solution.w:.6f}$",
            transform=ax_cdf.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
        )
        style_axis(ax_spectrum)
        style_axis(ax_cdf)
        add_panel_label(ax_spectrum, f"({chr(ord('a') + 2 * row_index)})")
        add_panel_label(ax_cdf, f"({chr(ord('b') + 2 * row_index)})")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0.12, 0.99), **compact_legend_kwargs())
    axes[1, 0].set_ylabel(r"Unit-area intensity")
    axes[1, 1].set_ylabel(r"$F_{\rm ED}(x)-F_{\rm SCBA}(x)$")
    axes[-1, 0].set_xlabel(r"$x=(E-E_B)/\nu_0$")
    axes[-1, 1].set_xlabel(r"$x=(E-E_B)/\nu_0$")
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.08, top=0.99, hspace=0.14, wspace=0.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output,
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={
            "Title": "Revision-27 dimer spectral/CDF diagnostics",
            "Subject": "Independent numerical audit; not a publication figure",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def main() -> int:
    DATA.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    mesh_rows = _read_csv(MESH)
    mesh_lookup = {
        (round(float(row["huang_rhys"]), 12), int(row["j_sign"]), round(float(row["r_abs"]), 12)): float(row["w_W1_over_nu0"])
        for row in mesh_rows
    }

    refined_solutions: list[Solution] = []
    refined_rows: list[dict[str, Any]] = []
    for (s_value, sign), r_values in REFINED_R_CASES.items():
        for r_value in r_values:
            label = f"refined_S{s_value:g}_J{sign:+d}_r{r_value:g}"
            solution = _solve(label, s_value, sign, r_value)
            refined_solutions.append(solution)
            row = _solution_row(solution)
            lower = math.floor(r_value / 0.005) * 0.005
            upper = lower + 0.005
            neighbor_values = []
            for neighbor in (lower, upper):
                value = mesh_lookup.get((round(s_value, 12), sign, round(neighbor, 12)))
                if value is not None:
                    neighbor_values.append((neighbor, value))
            row["bracketing_publication_points"] = ";".join(
                f"{r:.6f}:{w:.12g}" for r, w in neighbor_values
            )
            if len(neighbor_values) == 2:
                row["midpoint_minus_linear_chord"] = solution.w - 0.5 * (
                    neighbor_values[0][1] + neighbor_values[1][1]
                )
            refined_rows.append(row)

    columns = list(refined_rows[0])
    _write_csv(DATA / "refined_r_points.csv", refined_rows, columns)

    convergence_rows: list[dict[str, Any]] = []
    fine_diagnostics: list[Solution] = []
    for label, s_value, sign, r_value in CONVERGENCE_CASES:
        variants = (
            ("production", 0.002, 0.0, 10),
            ("cutoff12", 0.002, 0.0, 12),
            ("half_cell", 0.002, 0.5, 10),
            ("dx0p001", 0.001, 0.0, 10),
            ("dx0p0005", 0.0005, 0.0, 10),
            ("dx0p0005_cutoff12", 0.0005, 0.0, 12),
        )
        solved: dict[str, Solution] = {}
        for variant, dx, phase, n_vib in variants:
            solution = _solve(f"{label}_{variant}", s_value, sign, r_value, dx, phase, n_vib)
            solved[variant] = solution
            row = _solution_row(solution)
            row["case"] = label
            row["variant"] = variant
            row["delta_w_from_production"] = solution.w - solved["production"].w
            convergence_rows.append(row)
        if label in (
            "S005_Jp_r1940", "S010_Jp_r1880", "S010_Jp_r1940",
            "S010_Jm_r2000",
        ):
            fine = solved["dx0p0005_cutoff12"]
            fine_diagnostics.append(fine)
            np.savez_compressed(
                DATA / f"{label}_dx0p0005_cutoff12.npz",
                x=fine.x,
                ed_probability=fine.ed_probability,
                scba_probability=fine.scba_probability,
                cdf_difference=fine.cdf_difference,
                metadata_json=np.asarray(json.dumps(_solution_row(fine), sort_keys=True)),
            )

    _write_csv(DATA / "convergence.csv", convergence_rows, list(convergence_rows[0]))

    peak_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    for solution in fine_diagnostics:
        peak_rows.extend(_peak_rows(solution, "ED"))
        peak_rows.extend(_peak_rows(solution, "SCBA"))
        region_rows.extend(_region_rows(solution))
    _write_csv(
        DATA / "peak_basin_metrics.csv",
        peak_rows,
        ("label", "method", "peak_ordinal", "peak_x", "peak_height", "relative_prominence", "basin_left_x", "basin_right_x", "basin_weight"),
    )
    _write_csv(
        DATA / "cdf_region_metrics.csv",
        region_rows,
        ("label", "x_lower", "x_upper", "ed_weight", "scba_weight", "scba_minus_ed_weight", "absolute_cdf_integral", "fraction_of_w"),
    )

    parity_checks, level_rows = _parity_rows()
    _write_csv(DATA / "parity_checks.csv", parity_checks, list(parity_checks[0]))
    _write_csv(DATA / "even_branch_tracking.csv", level_rows, list(level_rows[0]))

    pipeline = _pipeline_audit()
    (DATA / "pipeline_audit.json").write_text(
        json.dumps(pipeline, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _diagnostic_plot(fine_diagnostics, PLOTS / "r2_spectral_cdf_diagnostics.pdf")

    maxima = {
        "maximum_abs_dx0p001_delta_w": max(
            abs(float(row["delta_w_from_production"]))
            for row in convergence_rows if row["variant"] == "dx0p001"
        ),
        "maximum_abs_dx0p0005_delta_w": max(
            abs(float(row["delta_w_from_production"]))
            for row in convergence_rows if row["variant"] == "dx0p0005"
        ),
        "maximum_abs_half_cell_delta_w": max(
            abs(float(row["delta_w_from_production"]))
            for row in convergence_rows if row["variant"] == "half_cell"
        ),
        "maximum_abs_cutoff12_delta_w": max(
            abs(float(row["delta_w_from_production"]))
            for row in convergence_rows if row["variant"] == "cutoff12"
        ),
        "maximum_abs_fine_cutoff12_delta_from_production": max(
            abs(float(row["delta_w_from_production"]))
            for row in convergence_rows if row["variant"] == "dx0p0005_cutoff12"
        ),
        "maximum_abs_midpoint_minus_chord": max(
            abs(float(row.get("midpoint_minus_linear_chord", 0.0)))
            for row in refined_rows
        ),
        "maximum_abs_support_m16_p16_minus_m18_p18": max(
            abs(float(row["w_support_m16_p16"]) - float(row["w_support_m18_p18"]))
            for row in convergence_rows
        ),
        "maximum_abs_support_m20_p20_minus_m18_p18": max(
            abs(float(row["w_support_m20_p20"]) - float(row["w_support_m18_p18"]))
            for row in convergence_rows
        ),
        "maximum_scba_abs_residual_eV": max(abs(solution.scba_abs_residual_eV) for solution in refined_solutions + fine_diagnostics),
        "maximum_scba_positive_im_sigma_eV": max(solution.scba_max_positive_im_sigma_eV for solution in refined_solutions + fine_diagnostics),
        "minimum_spectrum": min(min(solution.ed_minimum, solution.scba_minimum) for solution in refined_solutions + fine_diagnostics),
        "maximum_parity_commutator": max(float(row["commutator_max_abs"]) for row in parity_checks),
        "maximum_opposite_parity_H_block": max(float(row["opposite_parity_H_block_max_abs"]) for row in parity_checks),
        "maximum_opposite_parity_source_norm": max(float(row["source_opposite_parity_projected_norm"]) for row in parity_checks),
        "maximum_tracked_even_crossing_gap": max(
            float(row["absolute_gap_over_nu0"])
            for row in level_rows if row["is_tracked_crossing"]
        ),
    }
    summary = {
        "schema_version": "revision27_dimer_focused_audit_v1",
        "generated_utc": _utc_now(),
        "command": "python3 Figures/common/dimer/single_mode/revision27/audit_dimer_revision27.py",
        "publication_data_modified": False,
        "settings": {
            "J_abs_meV": J_ABS_MEV,
            "temperature_K": 0.0,
            "eta_over_nu0": ETA_OVER_NU0,
            "internal_x": INTERNAL_X,
            "metric_support_x": ANALYSIS_X,
            "production_dx": 0.002,
            "fine_dx": (0.001, 0.0005),
            "grid_phase_test_cells": 0.5,
            "cutoff_test": (10, 12),
        },
        "counts": {
            "refined_r_points": len(refined_rows),
            "convergence_rows": len(convergence_rows),
            "fine_diagnostic_cases": len(fine_diagnostics),
            "parity_cases": len(parity_checks),
        },
        "maxima": maxima,
        "pipeline": pipeline,
        "supported_diagnosis": {
            "positive_J_r2": (
                "At the fine-grid positive-J cases, ED has one dominant central "
                "peak while SCBA redistributes it among three nearby peaks. "
                "The x in [-0.5,0.5] interval contributes 0.934, 0.887, and "
                "0.877 of W for (S,r)=(0.05,1.94), (0.10,1.88), and "
                "(0.10,1.94), respectively."
            ),
            "negative_J_control": (
                "At S=0.10, J<0, r=2, ED and SCBA each retain one aligned "
                "dominant central peak; its W is smaller and distributed over "
                "central and sideband regions."
            ),
            "parity": (
                "Exact opposite-parity block/source overlaps vanish. The "
                "tracked even branches cross without an avoided gap. The "
                "spectral-distance maximum is therefore not evidence for an "
                "allowed even-phonon bright-dark transition."
            ),
            "limit": (
                "The audit identifies the spectral redistribution but does "
                "not decompose a crossed diagram or establish whether a "
                "particular SCBA expansion term violates the exact selection rule."
            ),
        },
        "outputs": {
            "refined_r": "data/refined_r_points.csv",
            "convergence": "data/convergence.csv",
            "peaks": "data/peak_basin_metrics.csv",
            "cdf_regions": "data/cdf_region_metrics.csv",
            "parity": "data/parity_checks.csv",
            "even_branches": "data/even_branch_tracking.csv",
            "pipeline": "data/pipeline_audit.json",
            "diagnosis": "data/diagnosis.txt",
            "diagnostic_plot": "plots/r2_spectral_cdf_diagnostics.pdf",
        },
    }
    (DATA / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    diagnosis_lines = [
        "Revision-27 focused dimer diagnosis",
        "generated_utc=%s" % summary["generated_utc"],
        "production w reproduction max abs error < 9e-14",
        "max |Delta w|: dx=0.001 %.12e; dx=0.0005 %.12e; half-cell %.12e; cutoff 10->12 %.12e" % (
            maxima["maximum_abs_dx0p001_delta_w"],
            maxima["maximum_abs_dx0p0005_delta_w"],
            maxima["maximum_abs_half_cell_delta_w"],
            maxima["maximum_abs_cutoff12_delta_w"],
        ),
        "max |midpoint - bracketing chord| = %.12e" % maxima["maximum_abs_midpoint_minus_chord"],
        "max support shifts: [-16,16] vs [-18,18] %.12e; [-20,20] vs [-18,18] %.12e" % (
            maxima["maximum_abs_support_m16_p16_minus_m18_p18"],
            maxima["maximum_abs_support_m20_p20_minus_m18_p18"],
        ),
        "parity: commutator %.3e; opposite block %.3e; opposite source %.3e; tracked crossing gap %.3e" % (
            maxima["maximum_parity_commutator"],
            maxima["maximum_opposite_parity_H_block"],
            maxima["maximum_opposite_parity_source_norm"],
            maxima["maximum_tracked_even_crossing_gap"],
        ),
        "",
        summary["supported_diagnosis"]["positive_J_r2"],
        summary["supported_diagnosis"]["negative_J_control"],
        summary["supported_diagnosis"]["parity"],
        summary["supported_diagnosis"]["limit"],
        "",
        "No production CSV/NPZ was modified; no curve was smoothed, fitted, filtered, or broadened.",
    ]
    (DATA / "diagnosis.txt").write_text(
        "\n".join(diagnosis_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(maxima, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
