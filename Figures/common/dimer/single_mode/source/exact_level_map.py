#!/usr/bin/env python3
"""Generate the exact difference-mode level and Rabi-parity map.

The symmetric single-mode Frenkel dimer factorizes into a displaced common
oscillator and the difference-mode quantum-Rabi Hamiltonian

    H_-/omega0 = (r/2) tau_z + b^dagger b
                 + sqrt(S/2) tau_x (b + b^dagger),

where ``r = 2 J / omega0``.  This script diagonalizes the two conserved
parity sectors separately, saves low exact levels and their ``|B,0>``
oscillator strengths, and validates the odd-phonon selection rule.  The
parity convention is

    Pi = tau_z (-1)^(b^dagger b),

so ``|B,0>`` and ``|D,m>`` have equal parity only for odd ``m``.

Only factorized difference-mode levels are generated here.  The analytically
displaced common-mode Franck--Condon ladder is deliberately not replicated in
the level map because it is independent of ``r`` and would obscure the
bright--dark avoided crossings.

The default physical mapping holds ``|J|=25 meV`` and reports
``omega0=2|J|/r``.  All diagonalization and validation are dimensionless, so
the data also apply to any simultaneous rescaling of ``J`` and ``omega0``.

The script is independent of the current working directory and writes only
to the dedicated ``results/frequency_scan`` and
``validation/frequency_scan`` subdirectories.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq, minimize_scalar


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from calculate_single_mode_ed import difference_mode_eigensystem  # noqa: E402
from single_mode_common import ModelParams, encode_metadata  # noqa: E402


SCHEMA_VERSION = 1
DEFAULT_RESULTS_DIR = MODULE_DIR / "results" / "frequency_scan"
DEFAULT_VALIDATION_DIR = MODULE_DIR / "validation" / "frequency_scan"

LEVEL_FIELDS = (
    "schema_version",
    "huang_rhys",
    "r_abs",
    "omega0_meV",
    "j_meV",
    "g_meV",
    "difference_coupling_V_meV",
    "n_vib",
    "level_index",
    "parity",
    "energy_minus_epsilon_v_over_omega0",
    "detuning_from_bare_bright_over_omega0",
    "difference_mode_energy_eV",
    "bright0_weight",
    "bright_electronic_population",
    "dark_electronic_population",
    "mean_difference_phonon",
    "dominant_electronic_state",
    "dominant_phonon_number",
    "dominant_basis_weight",
    "cutoff_edge_population",
    "eigen_residual",
)

CROSSING_FIELDS = (
    "schema_version",
    "crossing_type",
    "huang_rhys",
    "order_m",
    "bare_resonance_r",
    "parity_B0",
    "parity_Dm",
    "same_parity",
    "n_vib",
    "nominal_gap_over_omega0",
    "nominal_cluster_centroid_over_omega0",
    "nominal_B0_weighted_centroid_over_omega0",
    "nominal_B0_cluster_weight",
    "nominal_Dm_cluster_weight",
    "tracked_min_gap_over_omega0",
    "tracked_min_r",
    "tracked_min_cluster_centroid_over_omega0",
    "tracked_gap_ambiguous",
    "forward_B0_rank_at_nominal",
    "reverse_B0_rank_at_nominal",
    "forward_reverse_assignment_agree",
    "forward_min_overlap",
    "reverse_min_overlap",
    "target_subspace_weight_lower",
    "target_subspace_weight_upper",
    "direct_matrix_element_over_omega0",
    "one_phonon_V_over_omega0",
    "weak_two_level_gap_2V_over_omega0",
    "nominal_gap_over_2V",
    "exact_control_crossing_r",
    "exact_control_gap_over_omega0",
    "forbidden_block_coupling_over_omega0",
    "lower_rank",
    "upper_rank",
    "notes",
)

VALIDATION_FIELDS = (
    "schema_version",
    "test_name",
    "huang_rhys",
    "r_abs",
    "order_m",
    "n_vib",
    "observed",
    "expected",
    "absolute_error",
    "tolerance",
    "passed",
    "ambiguous",
    "notes",
)


@dataclass(frozen=True)
class SectorSolution:
    """One exact parity-sector eigensystem in the full basis."""

    parity: int
    energy: np.ndarray
    vector: np.ndarray


@dataclass(frozen=True)
class CutoffAudit:
    """Adaptive difference-mode cutoff and its worst comparison errors."""

    n_vib: int
    energy_error: float
    observable_error: float
    edge_population: float


def _parse_float_list(text: str) -> Tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(not np.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("expected a comma-separated finite float list")
    return values


def parity_diagonal(n_vib: int) -> np.ndarray:
    """Return diagonal entries of ``tau_z*(-1)^n`` in the B/D basis."""

    number_parity = np.where(np.arange(n_vib) % 2 == 0, 1, -1)
    return np.concatenate((number_parity, -number_parity)).astype(int)


def dimensionless_hamiltonian(
    huang_rhys: float,
    r_abs: float,
    n_vib: int,
) -> np.ndarray:
    """Build ``(H_- - epsilon_v I)/omega0`` for positive ``J``."""

    if huang_rhys < 0.0 or not np.isfinite(huang_rhys):
        raise ValueError("huang_rhys must be finite and nonnegative")
    if r_abs <= 0.0 or not np.isfinite(r_abs):
        raise ValueError("r_abs must be finite and positive")
    if n_vib < 2:
        raise ValueError("n_vib must be at least two")
    number = np.diag(np.arange(n_vib, dtype=float))
    raising = np.diag(np.sqrt(np.arange(1, n_vib, dtype=float)), 1)
    coordinate = raising + raising.T
    tau_z = np.diag((1.0, -1.0))
    tau_x = np.asarray(((0.0, 1.0), (1.0, 0.0)))
    return (
        np.kron(0.5 * r_abs * tau_z, np.eye(n_vib))
        + np.kron(np.eye(2), number)
        + np.sqrt(0.5 * huang_rhys) * np.kron(tau_x, coordinate)
    )


def solve_sector(
    huang_rhys: float,
    r_abs: float,
    n_vib: int,
    parity: int,
) -> SectorSolution:
    """Diagonalize a single exact Rabi-parity block."""

    if parity not in (-1, 1):
        raise ValueError("parity must be -1 or +1")
    hamiltonian = dimensionless_hamiltonian(huang_rhys, r_abs, n_vib)
    parity_values = parity_diagonal(n_vib)
    indices = np.flatnonzero(parity_values == parity)
    energy, sector_vector = eigh(
        hamiltonian[np.ix_(indices, indices)],
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    vector = np.zeros((2 * n_vib, len(indices)), dtype=float)
    vector[indices, :] = sector_vector
    return SectorSolution(parity, energy, vector)


def solve_parity_resolved(
    huang_rhys: float,
    r_abs: float,
    n_vib: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return energy-sorted levels, vectors, and exact parity labels."""

    positive = solve_sector(huang_rhys, r_abs, n_vib, +1)
    negative = solve_sector(huang_rhys, r_abs, n_vib, -1)
    energy = np.concatenate((positive.energy, negative.energy))
    vector = np.concatenate((positive.vector, negative.vector), axis=1)
    parity = np.concatenate(
        (
            np.ones(len(positive.energy), dtype=int),
            -np.ones(len(negative.energy), dtype=int),
        )
    )
    order = np.argsort(energy, kind="mergesort")
    return energy[order], vector[:, order], parity[order]


def _state_observables(
    vector: np.ndarray,
    n_vib: int,
) -> Dict[str, np.ndarray]:
    probability = np.square(np.abs(vector))
    bright_population = np.sum(probability[:n_vib, :], axis=0)
    dark_population = np.sum(probability[n_vib:, :], axis=0)
    number = np.arange(n_vib, dtype=float)
    mean_number = number @ (
        probability[:n_vib, :] + probability[n_vib:, :]
    )
    dominant = np.argmax(probability, axis=0)
    dominant_electronic = np.where(dominant < n_vib, 0, 1)
    dominant_number = dominant % n_vib
    dominant_weight = probability[dominant, np.arange(vector.shape[1])]
    edge_indices = np.asarray(
        (n_vib - 2, n_vib - 1, 2 * n_vib - 2, 2 * n_vib - 1)
    )
    edge_population = np.sum(probability[edge_indices, :], axis=0)
    return {
        "bright0_weight": probability[0, :],
        "bright_population": bright_population,
        "dark_population": dark_population,
        "mean_number": mean_number,
        "dominant_electronic": dominant_electronic,
        "dominant_number": dominant_number,
        "dominant_weight": dominant_weight,
        "edge_population": edge_population,
    }


def choose_cutoff(
    huang_rhys: float,
    r_checks: Sequence[float],
    n_levels: int,
    n_start: int,
    n_step: int,
    n_max: int,
    energy_tolerance: float,
    observable_tolerance: float,
    edge_tolerance: float,
) -> CutoffAudit:
    """Adapt ``N_vib`` using parity-sector levels and observables."""

    n_compare = int(np.ceil(n_levels / 2.0)) + 2
    for n_vib in range(n_start, n_max + 1, n_step):
        if n_vib + n_step > n_max:
            break
        worst_energy = 0.0
        worst_observable = 0.0
        worst_edge = 0.0
        for r_abs in r_checks:
            for parity in (+1, -1):
                small = solve_sector(huang_rhys, r_abs, n_vib, parity)
                large = solve_sector(huang_rhys, r_abs, n_vib + n_step, parity)
                count = min(n_compare, len(small.energy), len(large.energy))
                worst_energy = max(
                    worst_energy,
                    float(np.max(np.abs(small.energy[:count] - large.energy[:count]))),
                )
                small_obs = _state_observables(small.vector[:, :count], n_vib)
                large_obs = _state_observables(
                    large.vector[:, :count], n_vib + n_step
                )
                for key in ("bright0_weight", "bright_population", "mean_number"):
                    worst_observable = max(
                        worst_observable,
                        float(
                            np.max(
                                np.abs(small_obs[key][:count] - large_obs[key][:count])
                            )
                        ),
                    )
                worst_edge = max(
                    worst_edge,
                    float(np.max(small_obs["edge_population"][:count])),
                )
        if (
            worst_energy <= energy_tolerance
            and worst_observable <= observable_tolerance
            and worst_edge <= edge_tolerance
        ):
            return CutoffAudit(
                n_vib=n_vib,
                energy_error=worst_energy,
                observable_error=worst_observable,
                edge_population=worst_edge,
            )
    raise RuntimeError(
        "No converged cutoff for S=%g through N_vib=%d" % (huang_rhys, n_max)
    )


def build_r_grid(
    r_min: float,
    r_max: float,
    coarse_step: float,
    fine_step: float,
    fine_half_width: float,
) -> np.ndarray:
    """Construct a deterministic coarse grid with odd-resonance refinement."""

    if not (0.0 < r_min < r_max):
        raise ValueError("Require 0 < r_min < r_max")
    if min(coarse_step, fine_step, fine_half_width) <= 0.0:
        raise ValueError("Grid steps and fine half-width must be positive")
    values: List[float] = list(
        np.arange(r_min, r_max + 0.5 * coarse_step, coarse_step)
    )
    for resonance in (1.0, 3.0, 5.0):
        lower = max(r_min, resonance - fine_half_width)
        upper = min(r_max, resonance + fine_half_width)
        values.extend(np.arange(lower, upper + 0.5 * fine_step, fine_step))
    # ``arange`` need not land on either endpoint when ``r_min`` is not an
    # integer multiple of the coarse step.  Preserve the requested closed
    # interval explicitly; the endpoint rows are useful when judging whether
    # a tracked crossing has merely left the scanned range.
    values.extend((r_min, r_max, 1.0, 2.0, 3.0, 4.0, 5.0))
    rounded = np.unique(np.round(np.asarray(values, dtype=float), 12))
    return rounded[(rounded >= r_min) & (rounded <= r_max)]


def _odd_crossing(
    huang_rhys: float,
    order_m: int,
    n_vib: int,
    r_min: float,
    r_max: float,
) -> Dict[str, Any]:
    """Extract a parity-allowed odd crossing using fixed sector ranks."""

    if order_m % 2 != 1:
        raise ValueError("odd crossing requires odd order")
    center = solve_sector(huang_rhys, float(order_m), n_vib, +1)
    target_weight = (
        np.square(center.vector[0, :])
        + np.square(center.vector[n_vib + order_m, :])
    )
    pair = np.sort(np.argsort(target_weight)[-2:])
    lower_rank, upper_rank = int(pair[0]), int(pair[1])
    if upper_rank != lower_rank + 1:
        raise RuntimeError(
            "Target odd-crossing levels are not adjacent for S=%g, m=%d"
            % (huang_rhys, order_m)
        )
    nominal_gap = float(
        center.energy[upper_rank] - center.energy[lower_rank]
    )
    nominal_pair_energy = center.energy[pair]
    nominal_b0_weight = np.square(center.vector[0, pair])
    nominal_dm_weight = np.square(center.vector[n_vib + order_m, pair])
    nominal_cluster_centroid = float(np.mean(nominal_pair_energy))
    nominal_b0_cluster_weight = float(np.sum(nominal_b0_weight))
    nominal_b0_weighted_centroid = float(
        np.sum(nominal_pair_energy * nominal_b0_weight)
        / nominal_b0_cluster_weight
    )

    lower_bound = max(r_min, float(order_m) - 1.5)
    upper_bound = min(r_max, float(order_m) + 1.0)

    def fixed_rank_gap(r_value: float) -> float:
        solution = solve_sector(huang_rhys, float(r_value), n_vib, +1)
        return float(solution.energy[upper_rank] - solution.energy[lower_rank])

    minimum = minimize_scalar(
        fixed_rank_gap,
        bounds=(lower_bound, upper_bound),
        method="bounded",
        options={"xatol": 2.0e-12, "maxiter": 200},
    )
    at_minimum = solve_sector(huang_rhys, float(minimum.x), n_vib, +1)
    minimum_target_weight = (
        np.square(at_minimum.vector[0, :])
        + np.square(at_minimum.vector[n_vib + order_m, :])
    )
    boundary_distance = min(
        float(minimum.x) - lower_bound,
        upper_bound - float(minimum.x),
    )
    ambiguous = bool(
        boundary_distance < 2.0e-4
        or min(
            minimum_target_weight[lower_rank],
            minimum_target_weight[upper_rank],
        )
        < 0.45
    )
    coupling = np.sqrt(0.5 * huang_rhys)
    direct = coupling if order_m == 1 else 0.0

    def track_b0_rank(start: float, stop: float) -> Tuple[int, float]:
        span = abs(stop - start)
        count = max(3, int(np.ceil(span / 0.01)) + 1)
        track_grid = np.linspace(start, stop, count)
        first = solve_sector(huang_rhys, float(track_grid[0]), n_vib, +1)
        rank = int(np.argmax(np.square(first.vector[0, :])))
        previous = first.vector[:, rank]
        minimum_overlap = 1.0
        for r_value in track_grid[1:]:
            current = solve_sector(huang_rhys, float(r_value), n_vib, +1)
            overlaps = np.square(previous @ current.vector)
            rank = int(np.argmax(overlaps))
            minimum_overlap = min(minimum_overlap, float(overlaps[rank]))
            previous = current.vector[:, rank]
        return rank, minimum_overlap

    forward_rank, forward_overlap = track_b0_rank(
        lower_bound, float(order_m)
    )
    reverse_rank, reverse_overlap = track_b0_rank(
        upper_bound, float(order_m)
    )
    assignment_agrees = forward_rank == reverse_rank
    return {
        "schema_version": SCHEMA_VERSION,
        "crossing_type": "odd_allowed",
        "huang_rhys": huang_rhys,
        "order_m": order_m,
        "bare_resonance_r": float(order_m),
        "parity_B0": 1,
        "parity_Dm": 1,
        "same_parity": True,
        "n_vib": n_vib,
        "nominal_gap_over_omega0": nominal_gap,
        "nominal_cluster_centroid_over_omega0": nominal_cluster_centroid,
        "nominal_B0_weighted_centroid_over_omega0": nominal_b0_weighted_centroid,
        "nominal_B0_cluster_weight": nominal_b0_cluster_weight,
        "nominal_Dm_cluster_weight": float(np.sum(nominal_dm_weight)),
        "tracked_min_gap_over_omega0": float(minimum.fun),
        "tracked_min_r": float(minimum.x),
        "tracked_min_cluster_centroid_over_omega0": float(
            0.5
            * (
                at_minimum.energy[lower_rank]
                + at_minimum.energy[upper_rank]
            )
        ),
        "tracked_gap_ambiguous": ambiguous,
        "forward_B0_rank_at_nominal": forward_rank,
        "reverse_B0_rank_at_nominal": reverse_rank,
        "forward_reverse_assignment_agree": assignment_agrees,
        "forward_min_overlap": forward_overlap,
        "reverse_min_overlap": reverse_overlap,
        "target_subspace_weight_lower": float(
            minimum_target_weight[lower_rank]
        ),
        "target_subspace_weight_upper": float(
            minimum_target_weight[upper_rank]
        ),
        "direct_matrix_element_over_omega0": direct,
        "one_phonon_V_over_omega0": coupling if order_m == 1 else float("nan"),
        "weak_two_level_gap_2V_over_omega0": (
            2.0 * coupling if order_m == 1 else float("nan")
        ),
        "nominal_gap_over_2V": (
            nominal_gap / (2.0 * coupling)
            if order_m == 1 and coupling > 0.0
            else float("nan")
        ),
        "exact_control_crossing_r": float("nan"),
        "exact_control_gap_over_omega0": float("nan"),
        "forbidden_block_coupling_over_omega0": float("nan"),
        "lower_rank": lower_rank,
        "upper_rank": upper_rank,
        "notes": (
            "Same Rabi parity.  m=1 has bare off-diagonal V=g/sqrt(2); "
            "the weak isolated two-level separation is 2V=sqrt(2)g."
            if order_m == 1
            else "Same Rabi parity; the higher odd gap is mediated, not a direct matrix element."
        ),
    }


def _even_crossing(
    huang_rhys: float,
    order_m: int,
    n_vib: int,
    r_min: float,
    r_max: float,
) -> Dict[str, Any]:
    """Track the opposite-parity B0 and Dm branches at an even control.

    The target ranks are selected by their bare-basis overlaps at ``r=m``.
    Within a fixed parity sector an energy rank is an adiabatically continuous
    branch (generic same-parity levels avoid one another).  We therefore keep
    those two ranks fixed while searching the full requested r interval.  If
    they do not cross in that interval, no unrelated opposite-parity crossing
    is substituted: the root fields are NaN and the extraction is flagged.
    """

    if order_m % 2 != 0:
        raise ValueError("even control requires even order")
    nominal_positive = solve_sector(
        huang_rhys, float(order_m), n_vib, +1
    )
    nominal_negative = solve_sector(
        huang_rhys, float(order_m), n_vib, -1
    )
    positive_rank = int(np.argmax(np.square(nominal_positive.vector[0, :])))
    negative_rank = int(
        np.argmax(np.square(nominal_negative.vector[n_vib + order_m, :]))
    )
    nominal_positive_weight = float(
        np.square(nominal_positive.vector[0, positive_rank])
    )
    nominal_negative_weight = float(
        np.square(nominal_negative.vector[n_vib + order_m, negative_rank])
    )
    nominal_difference = float(
        nominal_positive.energy[positive_rank]
        - nominal_negative.energy[negative_rank]
    )

    sample_step = 0.01
    sample_count = max(3, int(np.ceil((r_max - r_min) / sample_step)) + 1)
    sample = np.linspace(r_min, r_max, sample_count)
    difference = np.asarray(
        [
            solve_sector(huang_rhys, value, n_vib, +1).energy[positive_rank]
            - solve_sector(huang_rhys, value, n_vib, -1).energy[negative_rank]
            for value in sample
        ]
    )
    roots: List[float] = []
    for index in range(len(sample) - 1):
        left = float(difference[index])
        right = float(difference[index + 1])
        if left == 0.0:
            roots.append(float(sample[index]))
        elif left * right < 0.0:
            roots.append(
                float(
                    brentq(
                        lambda value: (
                            solve_sector(
                                huang_rhys, value, n_vib, +1
                            ).energy[positive_rank]
                            - solve_sector(
                                huang_rhys, value, n_vib, -1
                            ).energy[negative_rank]
                        ),
                        float(sample[index]),
                        float(sample[index + 1]),
                        xtol=2.0e-14,
                        rtol=8.0 * np.finfo(float).eps,
                    )
                )
            )
    closest_index = int(np.argmin(np.abs(difference)))
    tracked_min_r = float(sample[closest_index])
    tracked_min_gap = abs(float(difference[closest_index]))
    if roots:
        root = min(roots, key=lambda value: abs(value - float(order_m)))
        positive = solve_sector(huang_rhys, root, n_vib, +1)
        negative = solve_sector(huang_rhys, root, n_vib, -1)
        gap = abs(
            float(
                positive.energy[positive_rank]
                - negative.energy[negative_rank]
            )
        )
        target_positive_weight = float(
            np.square(positive.vector[0, positive_rank])
        )
        target_negative_weight = float(
            np.square(negative.vector[n_vib + order_m, negative_rank])
        )
        tracked_min_r = root
        tracked_min_gap = gap
    else:
        root = float("nan")
        gap = float("nan")
        positive = nominal_positive
        negative = nominal_negative
        target_positive_weight = nominal_positive_weight
        target_negative_weight = nominal_negative_weight
    reference_r = root if np.isfinite(root) else float(order_m)
    hamiltonian = dimensionless_hamiltonian(
        huang_rhys, reference_r, n_vib
    )
    forbidden_coupling = abs(
        float(
            positive.vector[:, positive_rank]
            @ hamiltonian
            @ negative.vector[:, negative_rank]
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "crossing_type": "even_forbidden_control",
        "huang_rhys": huang_rhys,
        "order_m": order_m,
        "bare_resonance_r": float(order_m),
        "parity_B0": 1,
        "parity_Dm": -1,
        "same_parity": False,
        "n_vib": n_vib,
        "nominal_gap_over_omega0": abs(nominal_difference),
        "nominal_cluster_centroid_over_omega0": float(
            0.5
            * (
                nominal_positive.energy[positive_rank]
                + nominal_negative.energy[negative_rank]
            )
        ),
        "nominal_B0_weighted_centroid_over_omega0": float("nan"),
        "nominal_B0_cluster_weight": float("nan"),
        "nominal_Dm_cluster_weight": float("nan"),
        "tracked_min_gap_over_omega0": tracked_min_gap,
        "tracked_min_r": tracked_min_r,
        "tracked_min_cluster_centroid_over_omega0": (
            float(
                0.5
                * (
                    positive.energy[positive_rank]
                    + negative.energy[negative_rank]
                )
            )
            if roots
            else float("nan")
        ),
        "tracked_gap_ambiguous": not bool(roots),
        "forward_B0_rank_at_nominal": -1,
        "reverse_B0_rank_at_nominal": -1,
        "forward_reverse_assignment_agree": True,
        "forward_min_overlap": float("nan"),
        "reverse_min_overlap": float("nan"),
        "target_subspace_weight_lower": target_positive_weight,
        "target_subspace_weight_upper": target_negative_weight,
        "direct_matrix_element_over_omega0": 0.0,
        "one_phonon_V_over_omega0": float("nan"),
        "weak_two_level_gap_2V_over_omega0": float("nan"),
        "nominal_gap_over_2V": float("nan"),
        "exact_control_crossing_r": root,
        "exact_control_gap_over_omega0": gap,
        "forbidden_block_coupling_over_omega0": forbidden_coupling,
        "lower_rank": positive_rank,
        "upper_rank": negative_rank,
        "notes": (
            "B0 and Dm target ranks were selected at r=m and followed "
            "continuously within their separate parity blocks.  "
            + (
                "They cross exactly in the scanned range; this is not an avoided crossing.  "
                if roots
                else "No target-branch root occurs in the scanned range; no unrelated crossing was substituted.  "
            )
            + "A large spectral or SCBA error near this even control would not "
            "turn it into a symmetry-allowed exact B-D avoided crossing."
        ),
    }


def _validation_row(
    test_name: str,
    huang_rhys: float,
    r_abs: float,
    order_m: int,
    n_vib: int,
    observed: float,
    expected: float,
    tolerance: float,
    passed: bool,
    notes: str,
    ambiguous: bool = False,
) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "test_name": test_name,
        "huang_rhys": huang_rhys,
        "r_abs": r_abs,
        "order_m": order_m,
        "n_vib": n_vib,
        "observed": observed,
        "expected": expected,
        "absolute_error": abs(observed - expected),
        "tolerance": tolerance,
        "passed": bool(passed),
        "ambiguous": bool(ambiguous),
        "notes": notes,
    }


def run_validations(
    s_values: Sequence[float],
    cutoff: Dict[float, CutoffAudit],
    crossings: Sequence[Dict[str, Any]],
    n_levels: int,
    energy_tolerance: float,
    observable_tolerance: float,
    edge_tolerance: float,
    parity_tolerance: float,
    crossing_tolerance: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    crossing_lookup = {
        (float(row["huang_rhys"]), int(row["order_m"])): row
        for row in crossings
    }
    for huang_rhys in s_values:
        audit = cutoff[huang_rhys]
        rows.extend(
            (
                _validation_row(
                    "cutoff_energy",
                    huang_rhys,
                    float("nan"),
                    0,
                    audit.n_vib,
                    audit.energy_error,
                    0.0,
                    energy_tolerance,
                    audit.energy_error <= energy_tolerance,
                    "Worst parity-sector low-level error against N_vib+n_step.",
                ),
                _validation_row(
                    "cutoff_observable",
                    huang_rhys,
                    float("nan"),
                    0,
                    audit.n_vib,
                    audit.observable_error,
                    0.0,
                    observable_tolerance,
                    audit.observable_error <= observable_tolerance,
                    "Worst bright weight/population/mean-n change.",
                ),
                _validation_row(
                    "cutoff_edge_population",
                    huang_rhys,
                    float("nan"),
                    0,
                    audit.n_vib,
                    audit.edge_population,
                    0.0,
                    edge_tolerance,
                    audit.edge_population <= edge_tolerance,
                    "Worst population in the final two oscillator states.",
                ),
            )
        )
        for r_abs in (1.0, 2.0, 3.0, 4.0, 5.0):
            n_vib = audit.n_vib
            hamiltonian = dimensionless_hamiltonian(
                huang_rhys, r_abs, n_vib
            )
            parity = np.diag(parity_diagonal(n_vib).astype(float))
            commutator = hamiltonian @ parity - parity @ hamiltonian
            commutator_error = float(np.max(np.abs(commutator)))
            rows.append(
                _validation_row(
                    "rabi_parity_commutator",
                    huang_rhys,
                    r_abs,
                    int(round(r_abs)),
                    n_vib,
                    commutator_error,
                    0.0,
                    parity_tolerance,
                    commutator_error <= parity_tolerance,
                    "Pi=tau_z*(-1)^n commutes with the truncated Hamiltonian.",
                )
            )
            energy, vector, exact_parity = solve_parity_resolved(
                huang_rhys, r_abs, n_vib
            )
            count = min(n_levels, len(energy))
            residual = float(
                np.max(
                    np.abs(
                        hamiltonian @ vector[:, :count]
                        - vector[:, :count] * energy[:count][None, :]
                    )
                )
            )
            parity_expectation = np.sum(
                np.square(vector[:, :count])
                * parity_diagonal(n_vib)[:, None],
                axis=0,
            )
            parity_residual = float(
                np.max(np.abs(parity_expectation - exact_parity[:count]))
            )
            rows.append(
                _validation_row(
                    "eigen_residual",
                    huang_rhys,
                    r_abs,
                    int(round(r_abs)),
                    n_vib,
                    residual,
                    0.0,
                    5.0e-12,
                    residual <= 5.0e-12,
                    "Maximum componentwise residual of saved low levels.",
                )
            )
            rows.append(
                _validation_row(
                    "eigen_parity_residual",
                    huang_rhys,
                    r_abs,
                    int(round(r_abs)),
                    n_vib,
                    parity_residual,
                    0.0,
                    parity_tolerance,
                    parity_residual <= parity_tolerance,
                    "Separate block diagonalization prevents degenerate-state mixing.",
                )
            )

            omega0_meV = 50.0 / r_abs
            model = ModelParams(
                omega0_meV=omega0_meV,
                huang_rhys=huang_rhys,
                j_meV=25.0,
                temperature_K=0.0,
                eta_meV=0.01 * omega0_meV,
            )
            reference_energy, _ = difference_mode_eigensystem(model, n_vib)
            reference_dimensionless = (
                reference_energy - model.vertical_energy_eV
            ) / model.omega0_eV
            regression_error = float(
                np.max(
                    np.abs(reference_dimensionless[:count] - energy[:count])
                )
            )
            rows.append(
                _validation_row(
                    "production_ed_eigenvalue_regression",
                    huang_rhys,
                    r_abs,
                    int(round(r_abs)),
                    n_vib,
                    regression_error,
                    0.0,
                    2.0e-11,
                    regression_error <= 2.0e-11,
                    "Agrees with calculate_single_mode_ed.difference_mode_eigensystem.",
                )
            )

        for order_m in (1, 3, 5):
            crossing = crossing_lookup[(huang_rhys, order_m)]
            nominal_gap = float(crossing["nominal_gap_over_omega0"])
            rows.append(
                _validation_row(
                    "odd_parity_allowed_gap",
                    huang_rhys,
                    float(order_m),
                    order_m,
                    audit.n_vib,
                    nominal_gap,
                    float("nan"),
                    1.0e-9,
                    nominal_gap > 1.0e-9,
                    "Nonzero same-parity level repulsion at the bare odd resonance.",
                    bool(crossing["tracked_gap_ambiguous"]),
                )
            )
            if order_m == 1:
                hamiltonian = dimensionless_hamiltonian(
                    huang_rhys, 1.0, audit.n_vib
                )
                numerical_v = abs(float(hamiltonian[0, audit.n_vib + 1]))
                exact_v = float(np.sqrt(0.5 * huang_rhys))
                rows.append(
                    _validation_row(
                        "one_phonon_bare_matrix_element",
                        huang_rhys,
                        1.0,
                        1,
                        audit.n_vib,
                        numerical_v,
                        exact_v,
                        2.0e-14,
                        abs(numerical_v - exact_v) <= 2.0e-14,
                        "V=g/sqrt(2); the weak two-level avoided gap is 2V.",
                    )
                )
        for order_m in (2, 4):
            crossing = crossing_lookup[(huang_rhys, order_m)]
            gap = float(crossing["exact_control_gap_over_omega0"])
            coupling = float(
                crossing["forbidden_block_coupling_over_omega0"]
            )
            rows.append(
                _validation_row(
                    "even_bare_parity_product",
                    huang_rhys,
                    float(order_m),
                    order_m,
                    audit.n_vib,
                    float(crossing["parity_B0"])
                    * float(crossing["parity_Dm"]),
                    -1.0,
                    0.0,
                    int(crossing["parity_B0"])
                    * int(crossing["parity_Dm"])
                    == -1,
                    "B0 and even-Dm belong to opposite exact parity blocks.",
                )
            )
            if np.isfinite(gap):
                rows.append(
                    _validation_row(
                        "even_exact_target_crossing_gap",
                        huang_rhys,
                        float(crossing["exact_control_crossing_r"]),
                        order_m,
                        audit.n_vib,
                        gap,
                        0.0,
                        crossing_tolerance,
                        gap <= crossing_tolerance,
                        "Continuity-tracked opposite-parity target levels cross exactly.",
                    )
                )
            else:
                rows.append(
                    _validation_row(
                        "even_target_crossing_not_in_scan",
                        huang_rhys,
                        float(crossing["tracked_min_r"]),
                        order_m,
                        audit.n_vib,
                        float(crossing["tracked_min_gap_over_omega0"]),
                        float("nan"),
                        float("nan"),
                        True,
                        "No continuity-tracked B0/Dm root lies in the requested r range; parity still forbids hybridization.",
                        ambiguous=True,
                    )
                )
            rows.append(
                _validation_row(
                    "even_forbidden_block_coupling",
                    huang_rhys,
                    (
                        float(crossing["exact_control_crossing_r"])
                        if np.isfinite(
                            float(crossing["exact_control_crossing_r"])
                        )
                        else float(order_m)
                    ),
                    order_m,
                    audit.n_vib,
                    coupling,
                    0.0,
                    parity_tolerance,
                    coupling <= parity_tolerance,
                    "The full Hamiltonian has no matrix element between parity blocks.",
                )
            )
    return rows


def _atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_savez(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=path.stem + ".",
        suffix=".npz",
        dir=str(path.parent),
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(str(temporary), **arrays)
        os.replace(str(temporary), str(path))
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _check_outputs(paths: Sequence[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing exact-level-map outputs: %s; "
            "pass --overwrite to replace only these files"
            % ", ".join(str(path) for path in existing)
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s-values", type=_parse_float_list, default=(0.1, 0.5, 1.0))
    parser.add_argument("--j-abs-meV", type=float, default=25.0)
    parser.add_argument("--r-min", type=float, default=0.25)
    parser.add_argument("--r-max", type=float, default=5.5)
    parser.add_argument("--coarse-step", type=float, default=0.10)
    parser.add_argument("--fine-step", type=float, default=0.01)
    parser.add_argument("--fine-half-width", type=float, default=0.20)
    parser.add_argument("--n-levels", type=int, default=16)
    parser.add_argument("--n-vib-start", type=int, default=16)
    parser.add_argument("--n-vib-step", type=int, default=4)
    parser.add_argument("--n-vib-max", type=int, default=48)
    parser.add_argument("--energy-tolerance", type=float, default=5.0e-11)
    parser.add_argument("--observable-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--edge-tolerance", type=float, default=1.0e-9)
    parser.add_argument("--parity-tolerance", type=float, default=2.0e-13)
    parser.add_argument("--crossing-tolerance", type=float, default=2.0e-11)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.j_abs_meV <= 0.0 or not np.isfinite(args.j_abs_meV):
        raise ValueError("j-abs-meV must be finite and positive")
    if args.n_levels < 4:
        raise ValueError("n-levels must be at least four")
    if args.n_vib_start < args.n_levels // 2 + 3:
        raise ValueError("n-vib-start is too small for the requested level count")
    if args.n_vib_step < 1 or args.n_vib_max < args.n_vib_start + args.n_vib_step:
        raise ValueError("invalid adaptive cutoff range")
    s_values = tuple(sorted(set(float(value) for value in args.s_values)))
    if any(value <= 0.0 for value in s_values):
        raise ValueError("all S values must be positive")

    results_dir = args.results_dir
    validation_dir = args.validation_dir
    if not results_dir.is_absolute():
        results_dir = (MODULE_DIR / results_dir).resolve()
    if not validation_dir.is_absolute():
        validation_dir = (MODULE_DIR / validation_dir).resolve()
    level_csv = results_dir / "exact_level_map.csv"
    level_npz = results_dir / "exact_level_map.npz"
    crossing_csv = results_dir / "exact_level_map_crossings.csv"
    validation_csv = validation_dir / "exact_level_map_validation.csv"
    validation_npz = validation_dir / "exact_level_map_validation.npz"
    output_paths = (
        level_csv,
        level_npz,
        crossing_csv,
        validation_csv,
        validation_npz,
    )
    _check_outputs(output_paths, args.overwrite)

    base_r_grid = build_r_grid(
        args.r_min,
        args.r_max,
        args.coarse_step,
        args.fine_step,
        args.fine_half_width,
    )
    cutoff_checks = tuple(
        value
        for value in (args.r_min, 1.0, 2.0, 3.0, 4.0, 5.0, args.r_max)
        if args.r_min <= value <= args.r_max
    )
    cutoff: Dict[float, CutoffAudit] = {}
    for huang_rhys in s_values:
        cutoff[huang_rhys] = choose_cutoff(
            huang_rhys,
            cutoff_checks,
            args.n_levels,
            args.n_vib_start,
            args.n_vib_step,
            args.n_vib_max,
            args.energy_tolerance,
            args.observable_tolerance,
            args.edge_tolerance,
        )
        audit = cutoff[huang_rhys]
        print(
            "S=%g: N_vib=%d, dE=%.3e, dObs=%.3e, edge=%.3e"
            % (
                huang_rhys,
                audit.n_vib,
                audit.energy_error,
                audit.observable_error,
                audit.edge_population,
            )
        )

    crossings: List[Dict[str, Any]] = []
    for huang_rhys in s_values:
        n_vib = cutoff[huang_rhys].n_vib
        for order_m in (1, 3, 5):
            crossings.append(
                _odd_crossing(
                    huang_rhys,
                    order_m,
                    n_vib,
                    args.r_min,
                    args.r_max,
                )
            )
        for order_m in (2, 4):
            crossings.append(
                _even_crossing(
                    huang_rhys,
                    order_m,
                    n_vib,
                    args.r_min,
                    args.r_max,
                )
            )

    extra_r = []
    for row in crossings:
        for key in ("tracked_min_r", "exact_control_crossing_r"):
            value = float(row[key])
            if np.isfinite(value) and args.r_min <= value <= args.r_max:
                extra_r.append(value)
    r_grid = np.unique(
        np.round(np.concatenate((base_r_grid, np.asarray(extra_r))), 12)
    )

    shape = (len(s_values), len(r_grid), args.n_levels)
    energy = np.empty(shape)
    detuning = np.empty(shape)
    energy_eV = np.empty(shape)
    bright0_weight = np.empty(shape)
    bright_population = np.empty(shape)
    dark_population = np.empty(shape)
    mean_number = np.empty(shape)
    parity = np.empty(shape, dtype=np.int8)
    dominant_electronic = np.empty(shape, dtype=np.int8)
    dominant_number = np.empty(shape, dtype=np.int16)
    dominant_weight = np.empty(shape)
    edge_population = np.empty(shape)
    eigen_residual = np.empty(shape)
    level_rows: List[Dict[str, Any]] = []
    omega0_meV_grid = 2.0 * args.j_abs_meV / r_grid

    for s_index, huang_rhys in enumerate(s_values):
        n_vib = cutoff[huang_rhys].n_vib
        for r_index, r_abs in enumerate(r_grid):
            level_energy, level_vector, level_parity = solve_parity_resolved(
                huang_rhys, float(r_abs), n_vib
            )
            count = args.n_levels
            level_energy = level_energy[:count]
            level_vector = level_vector[:, :count]
            level_parity = level_parity[:count]
            obs = _state_observables(level_vector, n_vib)
            hamiltonian = dimensionless_hamiltonian(
                huang_rhys, float(r_abs), n_vib
            )
            residual_by_level = np.max(
                np.abs(
                    hamiltonian @ level_vector
                    - level_vector * level_energy[None, :]
                ),
                axis=0,
            )
            omega0_meV = float(omega0_meV_grid[r_index])
            omega0_eV = omega0_meV * 1.0e-3
            epsilon_v_eV = 2.0 + huang_rhys * omega0_eV
            physical_energy = epsilon_v_eV + omega0_eV * level_energy

            energy[s_index, r_index, :] = level_energy
            detuning[s_index, r_index, :] = level_energy - 0.5 * r_abs
            energy_eV[s_index, r_index, :] = physical_energy
            bright0_weight[s_index, r_index, :] = obs["bright0_weight"]
            bright_population[s_index, r_index, :] = obs["bright_population"]
            dark_population[s_index, r_index, :] = obs["dark_population"]
            mean_number[s_index, r_index, :] = obs["mean_number"]
            parity[s_index, r_index, :] = level_parity
            dominant_electronic[s_index, r_index, :] = obs[
                "dominant_electronic"
            ]
            dominant_number[s_index, r_index, :] = obs["dominant_number"]
            dominant_weight[s_index, r_index, :] = obs["dominant_weight"]
            edge_population[s_index, r_index, :] = obs["edge_population"]
            eigen_residual[s_index, r_index, :] = residual_by_level

            for level_index in range(count):
                level_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "huang_rhys": huang_rhys,
                        "r_abs": float(r_abs),
                        "omega0_meV": omega0_meV,
                        "j_meV": args.j_abs_meV,
                        "g_meV": omega0_meV * np.sqrt(huang_rhys),
                        "difference_coupling_V_meV": (
                            omega0_meV * np.sqrt(0.5 * huang_rhys)
                        ),
                        "n_vib": n_vib,
                        "level_index": level_index,
                        "parity": int(level_parity[level_index]),
                        "energy_minus_epsilon_v_over_omega0": float(
                            level_energy[level_index]
                        ),
                        "detuning_from_bare_bright_over_omega0": float(
                            level_energy[level_index] - 0.5 * r_abs
                        ),
                        "difference_mode_energy_eV": float(
                            physical_energy[level_index]
                        ),
                        "bright0_weight": float(obs["bright0_weight"][level_index]),
                        "bright_electronic_population": float(
                            obs["bright_population"][level_index]
                        ),
                        "dark_electronic_population": float(
                            obs["dark_population"][level_index]
                        ),
                        "mean_difference_phonon": float(
                            obs["mean_number"][level_index]
                        ),
                        "dominant_electronic_state": (
                            "B"
                            if int(obs["dominant_electronic"][level_index]) == 0
                            else "D"
                        ),
                        "dominant_phonon_number": int(
                            obs["dominant_number"][level_index]
                        ),
                        "dominant_basis_weight": float(
                            obs["dominant_weight"][level_index]
                        ),
                        "cutoff_edge_population": float(
                            obs["edge_population"][level_index]
                        ),
                        "eigen_residual": float(residual_by_level[level_index]),
                    }
                )

    validation_rows = run_validations(
        s_values,
        cutoff,
        crossings,
        args.n_levels,
        args.energy_tolerance,
        args.observable_tolerance,
        args.edge_tolerance,
        args.parity_tolerance,
        args.crossing_tolerance,
    )
    failures = [row for row in validation_rows if not bool(row["passed"])]

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "description": "Exact parity-resolved difference-mode level map",
        "hamiltonian": "(r/2) tau_z + n + sqrt(S/2) tau_x (b+bdagger)",
        "parity": "Pi=tau_z*(-1)^n",
        "oscillator_strength_definition": "|<eigenstate|B,0>|^2 in the difference mode",
        "common_mode_included": False,
        "j_abs_meV": args.j_abs_meV,
        "eta_over_omega0_context": 0.01,
        "temperature_K_context": 0.0,
        "s_values": list(s_values),
        "r_min": args.r_min,
        "r_max": args.r_max,
        "coarse_step": args.coarse_step,
        "fine_step": args.fine_step,
        "fine_half_width": args.fine_half_width,
        "n_levels": args.n_levels,
        "n_vib_by_s": {
            str(value): cutoff[value].n_vib for value in s_values
        },
        "tolerances": {
            "energy": args.energy_tolerance,
            "observable": args.observable_tolerance,
            "edge_population": args.edge_tolerance,
            "parity": args.parity_tolerance,
            "even_crossing_gap": args.crossing_tolerance,
        },
        "one_phonon_convention": (
            "V=g/sqrt(2); weak isolated two-level gap=2V=sqrt(2)g"
        ),
        "crossing_center_note": (
            "Odd/even integers are bare resonance controls; dressed crossing "
            "centers can shift, especially at moderate or strong S."
        ),
        "symmetry_breaking_note": (
            "Dipole asymmetry changes optical access but not H_- parity.  "
            "Hamiltonian site-energy, mode, or coupling asymmetry can mix the "
            "parity blocks and open formerly forbidden even gaps."
        ),
        "error_map_note": (
            "An SCBA spectral-error maximum at an even r control must not be "
            "identified with an exact B-D avoided crossing; H_- parity still "
            "forbids that crossing in the symmetric Hamiltonian."
        ),
        "full_spectrum_matching_note": (
            "A full optical feature additionally needs its common-mode replica "
            "index; this difference-mode level map intentionally has none."
        ),
        "validation_failure_count": len(failures),
    }

    _atomic_write_csv(level_csv, LEVEL_FIELDS, level_rows)
    _atomic_write_csv(crossing_csv, CROSSING_FIELDS, crossings)
    _atomic_write_csv(validation_csv, VALIDATION_FIELDS, validation_rows)
    _atomic_savez(
        level_npz,
        {
            "huang_rhys": np.asarray(s_values, dtype=float),
            "r_abs": r_grid,
            "omega0_meV": omega0_meV_grid,
            "j_abs_meV": np.asarray(args.j_abs_meV),
            "n_vib_by_s": np.asarray(
                [cutoff[value].n_vib for value in s_values], dtype=int
            ),
            "level_index": np.arange(args.n_levels, dtype=int),
            "energy_minus_epsilon_v_over_omega0": energy,
            "detuning_from_bare_bright_over_omega0": detuning,
            "difference_mode_energy_eV": energy_eV,
            "bright0_weight": bright0_weight,
            "bright_electronic_population": bright_population,
            "dark_electronic_population": dark_population,
            "mean_difference_phonon": mean_number,
            "parity": parity,
            "dominant_electronic_index": dominant_electronic,
            "dominant_phonon_number": dominant_number,
            "dominant_basis_weight": dominant_weight,
            "cutoff_edge_population": edge_population,
            "eigen_residual": eigen_residual,
            "metadata_json": encode_metadata(metadata),
        },
    )
    _atomic_savez(
        validation_npz,
        {
            "cutoff_huang_rhys": np.asarray(s_values, dtype=float),
            "cutoff_n_vib": np.asarray(
                [cutoff[value].n_vib for value in s_values], dtype=int
            ),
            "cutoff_energy_error": np.asarray(
                [cutoff[value].energy_error for value in s_values]
            ),
            "cutoff_observable_error": np.asarray(
                [cutoff[value].observable_error for value in s_values]
            ),
            "cutoff_edge_population": np.asarray(
                [cutoff[value].edge_population for value in s_values]
            ),
            "validation_passed": np.asarray(
                [bool(row["passed"]) for row in validation_rows], dtype=bool
            ),
            "metadata_json": encode_metadata(metadata),
        },
    )

    print("Saved %d level rows to %s" % (len(level_rows), level_csv))
    print("Saved %d crossing rows to %s" % (len(crossings), crossing_csv))
    print(
        "Saved %d validation rows to %s (%d failures)"
        % (len(validation_rows), validation_csv, len(failures))
    )
    if failures:
        for row in failures[:10]:
            print(
                "FAILED %s S=%s r=%s observed=%s tolerance=%s"
                % (
                    row["test_name"],
                    row["huang_rhys"],
                    row["r_abs"],
                    row["observed"],
                    row["tolerance"],
                )
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
