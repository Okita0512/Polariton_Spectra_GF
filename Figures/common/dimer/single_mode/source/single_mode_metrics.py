#!/usr/bin/env python3
"""Spectral metrics for the single-mode Frenkel-dimer benchmark.

The functions in this module operate on a common, uniformly spaced energy
grid.  Raw spectral areas are always retained.  Shape distances are evaluated
only after each spectrum has independently been normalized to unit area over
one explicitly declared finite window.

The spectra used by this project are ``-Im G^R`` rather than
``-Im G^R / pi``.  Consequently a normalized bright-state Green function has
an all-frequency raw area of pi.  Nothing in this module silently rescales a
raw area to pi or to one.

All energies are in eV.  The module depends only on NumPy and SciPy and does
not inspect or change the process working directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.stats import wasserstein_distance


@dataclass(frozen=True)
class WindowMoments:
    """Raw area and normalized finite-window moments of one spectrum."""

    window_min_eV: float
    window_max_eV: float
    raw_area: float
    moment_1_eV: float
    moment_2_eV2: float
    moment_3_eV3: float
    centroid_eV: float
    variance_eV2: float
    standard_deviation_eV: float
    third_central_moment_eV3: float
    skewness: float


@dataclass(frozen=True)
class SpectralComparison:
    """Raw sum-rule and unit-area shape metrics on one finite window."""

    window_min_eV: float
    window_max_eV: float
    reference_raw_area: float
    candidate_raw_area: float
    raw_area_difference_candidate_minus_reference: float
    raw_area_relative_difference: float
    unit_area_l1: float
    unit_area_l2_per_sqrt_eV: float
    bhattacharyya_coefficient: float
    wasserstein_eV: float
    reference_moments: WindowMoments
    candidate_moments: WindowMoments
    centroid_error_eV: float
    variance_error_eV2: float
    skewness_error: float


@dataclass(frozen=True)
class PeakRecord:
    """One resolved peak and the spectral basin assigned to it."""

    ordinal: int
    grid_index: int
    energy_eV: float
    height: float
    relative_height: float
    prominence: float
    relative_prominence: float
    prominence_left_base_eV: float
    prominence_right_base_eV: float
    basin_left_eV: float
    basin_right_eV: float
    basin_weight_raw: float
    basin_weight_fraction: float
    fwhm_eV: float
    fwhm_valid: bool
    fwhm_reason: str


@dataclass(frozen=True)
class PeakAnalysis:
    """Resolved peaks and a guarded dominant-peak/FWHM assessment."""

    window_min_eV: float
    window_max_eV: float
    prominence_fraction: float
    minimum_distance_eV: float
    dominance_height_ratio_required: float
    dominance_weight_fraction_required: float
    peaks: Tuple[PeakRecord, ...]
    dominant_ordinal: int
    dominant_energy_eV: float
    dominant_fwhm_eV: float
    dominant_fwhm_reliable: bool
    ambiguity_reason: str


@dataclass(frozen=True)
class PeakComparison:
    """Dominant-peak shift and guarded FWHM error for two spectra."""

    reference_resolved_peak_count: int
    candidate_resolved_peak_count: int
    reference_dominant_energy_eV: float
    candidate_dominant_energy_eV: float
    dominant_peak_shift_candidate_minus_reference_eV: float
    dominant_peak_shift_available: bool
    reference_dominant_fwhm_eV: float
    candidate_dominant_fwhm_eV: float
    fwhm_error_candidate_minus_reference_eV: float
    fwhm_error_reliable: bool
    ambiguity_reason: str


@dataclass(frozen=True)
class SidebandMatchRow:
    """A matched, missing, or spurious sideband."""

    status: str
    reference_ordinal: int
    candidate_ordinal: int
    reference_energy_eV: float
    candidate_energy_eV: float
    energy_shift_candidate_minus_reference_eV: float
    reference_weight: float
    candidate_weight: float
    weight_difference_candidate_minus_reference: float
    absolute_weight_error: float
    assignment_cost: float


@dataclass(frozen=True)
class SidebandMatchResult:
    """Peak-matching rows and aggregate position/weight errors."""

    method: str
    maximum_energy_difference_eV: float
    energy_cost_scale_eV: float
    weight_cost_factor: float
    unmatched_penalty: float
    matched_count: int
    missing_count: int
    spurious_count: int
    matched_energy_mae_eV: float
    matched_energy_rmse_eV: float
    matched_energy_max_abs_eV: float
    total_reference_weight: float
    total_candidate_weight: float
    aggregate_absolute_weight_error: float
    aggregate_relative_weight_error: float
    signed_total_weight_error_candidate_minus_reference: float
    rows: Tuple[SidebandMatchRow, ...]


@dataclass(frozen=True)
class WindowWeight:
    """Integrated spectral weight in a target window."""

    center_eV: float
    half_width_eV: float
    lower_eV: float
    upper_eV: float
    raw_weight: float
    analysis_raw_area: float
    fraction_of_analysis_area: float


@dataclass(frozen=True)
class ProjectedSelfEnergyAtPeak:
    """Operational projected Dyson self-energy evaluated at one peak."""

    energy_eV: float
    bare_projected_energy_eV: float
    eta_eV: float
    green_real_per_eV: float
    green_imag_per_eV: float
    sigma_real_eV: float
    sigma_imag_eV: float
    minus_two_imag_sigma_eV: float


def _as_real_vector(name: str, values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("{} must be a one-dimensional array".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains NaN or Inf".format(name))
    return array


def validate_uniform_grid(grid: Sequence[float]) -> Tuple[np.ndarray, float]:
    """Return a validated increasing uniform grid and its spacing."""

    energy = _as_real_vector("grid", grid)
    if energy.size < 3:
        raise ValueError("The common energy grid must contain at least 3 points")
    differences = np.diff(energy)
    if np.any(differences <= 0.0):
        raise ValueError("The common energy grid must be strictly increasing")
    spacing = float(np.median(differences))
    tolerance = max(
        64.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(energy)))),
        1.0e-9 * spacing,
    )
    if float(np.max(np.abs(differences - spacing))) > tolerance:
        raise ValueError("The common energy grid must be uniformly spaced")
    return energy, spacing


def _validate_spectrum(
    name: str,
    spectrum: Sequence[float],
    shape: Tuple[int, ...],
) -> np.ndarray:
    values = _as_real_vector(name, spectrum)
    if values.shape != shape:
        raise ValueError("{} and grid shapes differ".format(name))
    return values


def _resolve_window(
    grid: np.ndarray,
    window: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    if window is None:
        return float(grid[0]), float(grid[-1])
    if len(window) != 2:
        raise ValueError("window must contain (minimum_eV, maximum_eV)")
    lower, upper = float(window[0]), float(window[1])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("The analysis window must have finite ordered bounds")
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, abs(lower), abs(upper), abs(float(grid[0])), abs(float(grid[-1]))
    )
    if lower < grid[0] - tolerance or upper > grid[-1] + tolerance:
        raise ValueError("The analysis window lies outside the common grid")
    return max(lower, float(grid[0])), min(upper, float(grid[-1]))


def _slice_with_boundaries(
    grid: np.ndarray,
    arrays: Sequence[np.ndarray],
    lower: float,
    upper: float,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Restrict arrays to a window, interpolating both exact boundaries."""

    interior = (grid > lower) & (grid < upper)
    x = np.concatenate(([lower], grid[interior], [upper]))
    if x.size < 3:
        raise ValueError("The requested window contains fewer than 3 points")
    sliced: List[np.ndarray] = []
    for values in arrays:
        if np.iscomplexobj(values):
            left = np.interp(lower, grid, values.real) + 1.0j * np.interp(
                lower, grid, values.imag
            )
            right = np.interp(upper, grid, values.real) + 1.0j * np.interp(
                upper, grid, values.imag
            )
        else:
            left = np.interp(lower, grid, values)
            right = np.interp(upper, grid, values)
        sliced.append(np.concatenate(([left], values[interior], [right])))
    return x, sliced


def _trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    weights = np.empty_like(grid)
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return weights


def _nonnegative_for_shape_metrics(
    name: str,
    values: np.ndarray,
    relative_tolerance: float,
) -> np.ndarray:
    if relative_tolerance < 0.0:
        raise ValueError("negative_tolerance must be nonnegative")
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    minimum = float(np.min(values))
    if minimum < -relative_tolerance * scale:
        raise ValueError(
            "{} has materially negative spectral weight ({:.6e})".format(
                name, minimum
            )
        )
    return np.maximum(values, 0.0)


def raw_area(
    grid: Sequence[float],
    spectrum: Sequence[float],
    window: Optional[Tuple[float, float]] = None,
) -> float:
    """Integrate an unscaled spectrum over the declared finite window."""

    energy, _ = validate_uniform_grid(grid)
    values = _validate_spectrum("spectrum", spectrum, energy.shape)
    lower, upper = _resolve_window(energy, window)
    x, sliced = _slice_with_boundaries(energy, [values], lower, upper)
    return float(np.trapz(sliced[0], x))


def finite_window_moments(
    grid: Sequence[float],
    spectrum: Sequence[float],
    window: Optional[Tuple[float, float]] = None,
    negative_tolerance: float = 1.0e-10,
) -> WindowMoments:
    """Return first three raw and central moments on a finite window.

    Tiny negative values within ``negative_tolerance`` of the spectral scale
    are clipped for probability moments.  The reported ``raw_area`` is always
    evaluated before clipping so causality/sum-rule diagnostics remain visible.
    """

    energy, _ = validate_uniform_grid(grid)
    values = _validate_spectrum("spectrum", spectrum, energy.shape)
    lower, upper = _resolve_window(energy, window)
    x, sliced = _slice_with_boundaries(energy, [values], lower, upper)
    raw_values = sliced[0]
    area_raw = float(np.trapz(raw_values, x))
    physical = _nonnegative_for_shape_metrics(
        "spectrum", raw_values, negative_tolerance
    )
    area = float(np.trapz(physical, x))
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("Finite-window moments require positive spectral area")
    probability = physical / area
    moment_1 = float(np.trapz(x * probability, x))
    moment_2 = float(np.trapz(x**2 * probability, x))
    moment_3 = float(np.trapz(x**3 * probability, x))
    centered = x - moment_1
    variance = max(0.0, float(np.trapz(centered**2 * probability, x)))
    third_central = float(np.trapz(centered**3 * probability, x))
    standard_deviation = float(np.sqrt(variance))
    variance_floor = 256.0 * np.finfo(float).eps * max(
        1.0, moment_1**2
    )
    skewness = (
        third_central / variance**1.5
        if variance > variance_floor
        else float("nan")
    )
    return WindowMoments(
        window_min_eV=lower,
        window_max_eV=upper,
        raw_area=area_raw,
        moment_1_eV=moment_1,
        moment_2_eV2=moment_2,
        moment_3_eV3=moment_3,
        centroid_eV=moment_1,
        variance_eV2=variance,
        standard_deviation_eV=standard_deviation,
        third_central_moment_eV3=third_central,
        skewness=float(skewness),
    )


def compare_spectra(
    grid: Sequence[float],
    reference: Sequence[float],
    candidate: Sequence[float],
    window: Optional[Tuple[float, float]] = None,
    negative_tolerance: float = 1.0e-10,
) -> SpectralComparison:
    """Compare raw areas and unit-area shapes on the same uniform grid."""

    energy, _ = validate_uniform_grid(grid)
    reference_values = _validate_spectrum(
        "reference", reference, energy.shape
    )
    candidate_values = _validate_spectrum(
        "candidate", candidate, energy.shape
    )
    lower, upper = _resolve_window(energy, window)
    x, sliced = _slice_with_boundaries(
        energy, [reference_values, candidate_values], lower, upper
    )
    reference_raw, candidate_raw = sliced
    reference_area = float(np.trapz(reference_raw, x))
    candidate_area = float(np.trapz(candidate_raw, x))
    if reference_area == 0.0:
        relative_area_error = float("nan")
    else:
        relative_area_error = (candidate_area - reference_area) / reference_area

    reference_positive = _nonnegative_for_shape_metrics(
        "reference", reference_raw, negative_tolerance
    )
    candidate_positive = _nonnegative_for_shape_metrics(
        "candidate", candidate_raw, negative_tolerance
    )
    reference_positive_area = float(np.trapz(reference_positive, x))
    candidate_positive_area = float(np.trapz(candidate_positive, x))
    if reference_positive_area <= 0.0 or candidate_positive_area <= 0.0:
        raise ValueError("Shape metrics require two positive spectral areas")
    reference_probability = reference_positive / reference_positive_area
    candidate_probability = candidate_positive / candidate_positive_area
    difference = candidate_probability - reference_probability
    l1 = float(np.trapz(np.abs(difference), x))
    l2 = float(np.sqrt(np.trapz(difference**2, x)))
    bhattacharyya = float(
        np.trapz(
            np.sqrt(reference_probability * candidate_probability), x
        )
    )
    # scipy.stats.wasserstein_distance expects point masses.  Trapezoid
    # quadrature weights preserve the continuum normalization, including when
    # the requested window cuts through the first or last grid interval.
    quadrature = _trapezoid_weights(x)
    wasserstein = float(
        wasserstein_distance(
            x,
            x,
            u_weights=reference_probability * quadrature,
            v_weights=candidate_probability * quadrature,
        )
    )

    reference_moments = finite_window_moments(
        energy,
        reference_values,
        (lower, upper),
        negative_tolerance=negative_tolerance,
    )
    candidate_moments = finite_window_moments(
        energy,
        candidate_values,
        (lower, upper),
        negative_tolerance=negative_tolerance,
    )
    return SpectralComparison(
        window_min_eV=lower,
        window_max_eV=upper,
        reference_raw_area=reference_area,
        candidate_raw_area=candidate_area,
        raw_area_difference_candidate_minus_reference=(
            candidate_area - reference_area
        ),
        raw_area_relative_difference=float(relative_area_error),
        unit_area_l1=l1,
        unit_area_l2_per_sqrt_eV=l2,
        bhattacharyya_coefficient=bhattacharyya,
        wasserstein_eV=wasserstein,
        reference_moments=reference_moments,
        candidate_moments=candidate_moments,
        centroid_error_eV=(
            candidate_moments.centroid_eV - reference_moments.centroid_eV
        ),
        variance_error_eV2=(
            candidate_moments.variance_eV2 - reference_moments.variance_eV2
        ),
        skewness_error=(
            candidate_moments.skewness - reference_moments.skewness
        ),
    )


def _crossing_energy(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    target: float,
) -> float:
    if y1 == y0:
        return 0.5 * (x0 + x1)
    fraction = (target - y0) / (y1 - y0)
    return float(x0 + fraction * (x1 - x0))


def _peak_fwhm(
    x: np.ndarray,
    values: np.ndarray,
    peak_index: int,
    basin_left: int,
    basin_right: int,
) -> Tuple[float, bool, str]:
    half_height = 0.5 * float(values[peak_index])
    left_crossing: Optional[float] = None
    for index in range(peak_index - 1, basin_left - 1, -1):
        if values[index] <= half_height <= values[index + 1]:
            left_crossing = _crossing_energy(
                float(x[index]),
                float(values[index]),
                float(x[index + 1]),
                float(values[index + 1]),
                half_height,
            )
            break
    if left_crossing is None:
        return float("nan"), False, "no_left_half_maximum_crossing"

    right_crossing: Optional[float] = None
    for index in range(peak_index, basin_right):
        if values[index] >= half_height >= values[index + 1]:
            right_crossing = _crossing_energy(
                float(x[index]),
                float(values[index]),
                float(x[index + 1]),
                float(values[index + 1]),
                half_height,
            )
            break
    if right_crossing is None:
        return float("nan"), False, "no_right_half_maximum_crossing"
    width = right_crossing - left_crossing
    if not np.isfinite(width) or width <= 0.0:
        return float("nan"), False, "nonpositive_half_maximum_width"
    return float(width), True, ""


def analyze_peaks(
    grid: Sequence[float],
    spectrum: Sequence[float],
    window: Optional[Tuple[float, float]] = None,
    prominence_fraction: float = 0.02,
    minimum_distance_eV: float = 0.0,
    dominance_height_ratio: float = 1.5,
    dominance_weight_fraction: float = 0.50,
    negative_tolerance: float = 1.0e-10,
) -> PeakAnalysis:
    """Resolve peaks, basin weights, and a guarded dominant-peak FWHM.

    A dominant FWHM is called reliable only when half-maximum crossings exist
    inside its basin and either it is the sole resolved peak or it passes both
    explicitly supplied dominance thresholds.
    """

    if not 0.0 < prominence_fraction <= 1.0:
        raise ValueError("prominence_fraction must lie in (0, 1]")
    if minimum_distance_eV < 0.0:
        raise ValueError("minimum_distance_eV must be nonnegative")
    if dominance_height_ratio < 1.0:
        raise ValueError("dominance_height_ratio must be at least one")
    if not 0.0 <= dominance_weight_fraction <= 1.0:
        raise ValueError("dominance_weight_fraction must lie in [0, 1]")

    energy, spacing = validate_uniform_grid(grid)
    raw_values = _validate_spectrum("spectrum", spectrum, energy.shape)
    lower, upper = _resolve_window(energy, window)
    x, sliced = _slice_with_boundaries(energy, [raw_values], lower, upper)
    values = _nonnegative_for_shape_metrics(
        "spectrum", sliced[0], negative_tolerance
    )
    maximum = float(np.max(values))
    distance_samples = max(1, int(np.ceil(minimum_distance_eV / spacing)))
    if maximum <= 0.0:
        return PeakAnalysis(
            lower,
            upper,
            prominence_fraction,
            minimum_distance_eV,
            dominance_height_ratio,
            dominance_weight_fraction,
            tuple(),
            0,
            float("nan"),
            float("nan"),
            False,
            "no_positive_spectral_peak",
        )

    peak_indices, properties = find_peaks(
        values,
        prominence=prominence_fraction * maximum,
        distance=distance_samples,
    )
    if peak_indices.size == 0:
        return PeakAnalysis(
            lower,
            upper,
            prominence_fraction,
            minimum_distance_eV,
            dominance_height_ratio,
            dominance_weight_fraction,
            tuple(),
            0,
            float("nan"),
            float("nan"),
            False,
            "no_interior_peak_meets_prominence",
        )

    # Basin boundaries are the lowest sampled valleys between consecutive
    # resolved peaks.  The common valley belongs to both closed integrations,
    # but no energy interval is counted twice.
    boundaries: List[int] = [0]
    for left_peak, right_peak in zip(peak_indices[:-1], peak_indices[1:]):
        segment = values[left_peak : right_peak + 1]
        boundaries.append(int(left_peak + np.argmin(segment)))
    boundaries.append(len(x) - 1)
    total_weight = float(np.trapz(values, x))
    records: List[PeakRecord] = []
    for peak_number, peak_index in enumerate(peak_indices):
        basin_left = boundaries[peak_number]
        basin_right = boundaries[peak_number + 1]
        basin_weight = float(
            np.trapz(
                values[basin_left : basin_right + 1],
                x[basin_left : basin_right + 1],
            )
        )
        width, width_valid, width_reason = _peak_fwhm(
            x, values, int(peak_index), basin_left, basin_right
        )
        records.append(
            PeakRecord(
                ordinal=peak_number + 1,
                grid_index=int(
                    np.argmin(np.abs(energy - float(x[peak_index])))
                ),
                energy_eV=float(x[peak_index]),
                height=float(values[peak_index]),
                relative_height=float(values[peak_index] / maximum),
                prominence=float(properties["prominences"][peak_number]),
                relative_prominence=float(
                    properties["prominences"][peak_number] / maximum
                ),
                prominence_left_base_eV=float(
                    x[int(properties["left_bases"][peak_number])]
                ),
                prominence_right_base_eV=float(
                    x[int(properties["right_bases"][peak_number])]
                ),
                basin_left_eV=float(x[basin_left]),
                basin_right_eV=float(x[basin_right]),
                basin_weight_raw=basin_weight,
                basin_weight_fraction=(
                    basin_weight / total_weight
                    if total_weight > 0.0
                    else float("nan")
                ),
                fwhm_eV=width,
                fwhm_valid=width_valid,
                fwhm_reason=width_reason,
            )
        )

    dominant_position = int(
        np.argmax(np.asarray([record.height for record in records]))
    )
    dominant = records[dominant_position]
    reasons: List[str] = []
    if not dominant.fwhm_valid:
        reasons.append(dominant.fwhm_reason)
    if len(records) > 1:
        ordered_heights = sorted(
            (record.height for record in records), reverse=True
        )
        ratio = (
            ordered_heights[0] / ordered_heights[1]
            if ordered_heights[1] > 0.0
            else float("inf")
        )
        if ratio < dominance_height_ratio:
            reasons.append("competing_peak_height")
        if dominant.basin_weight_fraction < dominance_weight_fraction:
            reasons.append("dominant_basin_weight_too_small")
    reliable = len(reasons) == 0
    return PeakAnalysis(
        window_min_eV=lower,
        window_max_eV=upper,
        prominence_fraction=prominence_fraction,
        minimum_distance_eV=minimum_distance_eV,
        dominance_height_ratio_required=dominance_height_ratio,
        dominance_weight_fraction_required=dominance_weight_fraction,
        peaks=tuple(records),
        dominant_ordinal=dominant.ordinal,
        dominant_energy_eV=dominant.energy_eV,
        dominant_fwhm_eV=(
            dominant.fwhm_eV if reliable else float("nan")
        ),
        dominant_fwhm_reliable=reliable,
        ambiguity_reason=";".join(reasons),
    )


def compare_peak_analyses(
    reference: PeakAnalysis,
    candidate: PeakAnalysis,
) -> PeakComparison:
    """Compare dominant peaks without inventing an ambiguous FWHM error."""

    reference_has_peak = len(reference.peaks) > 0
    candidate_has_peak = len(candidate.peaks) > 0
    shift_available = reference_has_peak and candidate_has_peak
    shift = (
        candidate.dominant_energy_eV - reference.dominant_energy_eV
        if shift_available
        else float("nan")
    )
    fwhm_reliable = (
        reference.dominant_fwhm_reliable
        and candidate.dominant_fwhm_reliable
    )
    fwhm_error = (
        candidate.dominant_fwhm_eV - reference.dominant_fwhm_eV
        if fwhm_reliable
        else float("nan")
    )
    reasons: List[str] = []
    if not reference_has_peak:
        reasons.append("reference_has_no_resolved_peak")
    elif not reference.dominant_fwhm_reliable:
        reasons.append(
            "reference_fwhm_ambiguous:{}".format(
                reference.ambiguity_reason or "unspecified"
            )
        )
    if not candidate_has_peak:
        reasons.append("candidate_has_no_resolved_peak")
    elif not candidate.dominant_fwhm_reliable:
        reasons.append(
            "candidate_fwhm_ambiguous:{}".format(
                candidate.ambiguity_reason or "unspecified"
            )
        )
    return PeakComparison(
        reference_resolved_peak_count=len(reference.peaks),
        candidate_resolved_peak_count=len(candidate.peaks),
        reference_dominant_energy_eV=(
            reference.dominant_energy_eV
            if reference_has_peak
            else float("nan")
        ),
        candidate_dominant_energy_eV=(
            candidate.dominant_energy_eV
            if candidate_has_peak
            else float("nan")
        ),
        dominant_peak_shift_candidate_minus_reference_eV=float(shift),
        dominant_peak_shift_available=shift_available,
        reference_dominant_fwhm_eV=(
            reference.dominant_fwhm_eV
            if reference.dominant_fwhm_reliable
            else float("nan")
        ),
        candidate_dominant_fwhm_eV=(
            candidate.dominant_fwhm_eV
            if candidate.dominant_fwhm_reliable
            else float("nan")
        ),
        fwhm_error_candidate_minus_reference_eV=float(fwhm_error),
        fwhm_error_reliable=fwhm_reliable,
        ambiguity_reason=";".join(reasons),
    )


def _peak_weight(peak: PeakRecord, use_fractional_weights: bool) -> float:
    return (
        float(peak.basin_weight_fraction)
        if use_fractional_weights
        else float(peak.basin_weight_raw)
    )


def _pair_costs(
    reference: Sequence[PeakRecord],
    candidate: Sequence[PeakRecord],
    maximum_energy_difference_eV: float,
    energy_cost_scale_eV: float,
    weight_cost_factor: float,
    use_fractional_weights: bool,
) -> np.ndarray:
    costs = np.full((len(reference), len(candidate)), np.inf, dtype=float)
    for ref_index, ref_peak in enumerate(reference):
        for candidate_index, candidate_peak in enumerate(candidate):
            energy_error = abs(candidate_peak.energy_eV - ref_peak.energy_eV)
            if energy_error <= maximum_energy_difference_eV:
                weight_error = abs(
                    _peak_weight(candidate_peak, use_fractional_weights)
                    - _peak_weight(ref_peak, use_fractional_weights)
                )
                costs[ref_index, candidate_index] = (
                    energy_error / energy_cost_scale_eV
                    + weight_cost_factor * weight_error
                )
    return costs


def _monotone_assignment(
    costs: np.ndarray,
    unmatched_penalty: float,
) -> List[Tuple[str, int, int, float]]:
    n_reference, n_candidate = costs.shape
    score = np.full((n_reference + 1, n_candidate + 1), np.inf)
    operation = np.full((n_reference + 1, n_candidate + 1), "", dtype=object)
    score[0, 0] = 0.0
    for ref_index in range(1, n_reference + 1):
        score[ref_index, 0] = ref_index * unmatched_penalty
        operation[ref_index, 0] = "missing"
    for candidate_index in range(1, n_candidate + 1):
        score[0, candidate_index] = candidate_index * unmatched_penalty
        operation[0, candidate_index] = "spurious"

    for ref_index in range(1, n_reference + 1):
        for candidate_index in range(1, n_candidate + 1):
            choices: List[Tuple[float, int, str]] = [
                (
                    score[ref_index - 1, candidate_index]
                    + unmatched_penalty,
                    1,
                    "missing",
                ),
                (
                    score[ref_index, candidate_index - 1]
                    + unmatched_penalty,
                    2,
                    "spurious",
                ),
            ]
            pair_cost = costs[ref_index - 1, candidate_index - 1]
            if np.isfinite(pair_cost):
                choices.append(
                    (
                        score[ref_index - 1, candidate_index - 1]
                        + pair_cost,
                        0,
                        "matched",
                    )
                )
            best = min(choices, key=lambda item: (item[0], item[1]))
            score[ref_index, candidate_index] = best[0]
            operation[ref_index, candidate_index] = best[2]

    assignments: List[Tuple[str, int, int, float]] = []
    ref_index, candidate_index = n_reference, n_candidate
    while ref_index > 0 or candidate_index > 0:
        chosen = operation[ref_index, candidate_index]
        if chosen == "matched":
            assignments.append(
                (
                    chosen,
                    ref_index - 1,
                    candidate_index - 1,
                    float(costs[ref_index - 1, candidate_index - 1]),
                )
            )
            ref_index -= 1
            candidate_index -= 1
        elif chosen == "missing":
            assignments.append(
                (chosen, ref_index - 1, -1, float(unmatched_penalty))
            )
            ref_index -= 1
        elif chosen == "spurious":
            assignments.append(
                (chosen, -1, candidate_index - 1, float(unmatched_penalty))
            )
            candidate_index -= 1
        else:
            raise RuntimeError("Monotone sideband assignment backtrace failed")
    assignments.reverse()
    return assignments


def _hungarian_assignment(
    costs: np.ndarray,
    unmatched_penalty: float,
) -> List[Tuple[str, int, int, float]]:
    n_reference, n_candidate = costs.shape
    if n_reference == 0:
        return [
            ("spurious", -1, index, float(unmatched_penalty))
            for index in range(n_candidate)
        ]
    if n_candidate == 0:
        return [
            ("missing", index, -1, float(unmatched_penalty))
            for index in range(n_reference)
        ]

    dimension = n_reference + n_candidate
    finite_costs = costs[np.isfinite(costs)]
    scale = max(
        1.0,
        unmatched_penalty,
        float(np.max(finite_costs)) if finite_costs.size else 1.0,
    )
    forbidden = 1.0e6 * dimension * scale
    augmented = np.full((dimension, dimension), forbidden, dtype=float)
    augmented[:n_reference, :n_candidate] = np.where(
        np.isfinite(costs), costs, forbidden
    )
    for ref_index in range(n_reference):
        augmented[ref_index, n_candidate + ref_index] = unmatched_penalty
    for candidate_index in range(n_candidate):
        augmented[n_reference + candidate_index, candidate_index] = (
            unmatched_penalty
        )
    augmented[n_reference:, n_candidate:] = 0.0

    row_indices, column_indices = linear_sum_assignment(augmented)
    assignments: List[Tuple[str, int, int, float]] = []
    for row, column in zip(row_indices, column_indices):
        if row < n_reference and column < n_candidate:
            if augmented[row, column] >= forbidden:
                raise RuntimeError("Hungarian assignment selected a forbidden pair")
            assignments.append(
                ("matched", int(row), int(column), float(costs[row, column]))
            )
        elif row < n_reference and column >= n_candidate:
            assignments.append(
                ("missing", int(row), -1, float(unmatched_penalty))
            )
        elif row >= n_reference and column < n_candidate:
            assignments.append(
                ("spurious", -1, int(column), float(unmatched_penalty))
            )
    return assignments


def match_sidebands(
    reference_peaks: Sequence[PeakRecord],
    candidate_peaks: Sequence[PeakRecord],
    maximum_energy_difference_eV: float,
    method: str = "monotone",
    energy_cost_scale_eV: Optional[float] = None,
    weight_cost_factor: float = 0.25,
    unmatched_penalty: float = 1.0,
    use_fractional_weights: bool = True,
) -> SidebandMatchResult:
    """Match resolved sidebands with explicit missing/spurious output rows.

    ``method='monotone'`` uses sequence alignment and preserves spectral order.
    ``method='hungarian'`` permits a general minimum-cost assignment.  Pair
    costs contain an energy term and an optional basin-weight term.  Pairs
    farther apart than ``maximum_energy_difference_eV`` are forbidden.
    """

    if maximum_energy_difference_eV <= 0.0:
        raise ValueError("maximum_energy_difference_eV must be positive")
    if energy_cost_scale_eV is None:
        energy_cost_scale_eV = maximum_energy_difference_eV
    if energy_cost_scale_eV <= 0.0:
        raise ValueError("energy_cost_scale_eV must be positive")
    if weight_cost_factor < 0.0:
        raise ValueError("weight_cost_factor must be nonnegative")
    if unmatched_penalty <= 0.0:
        raise ValueError("unmatched_penalty must be positive")
    if method not in ("monotone", "hungarian"):
        raise ValueError("method must be 'monotone' or 'hungarian'")

    reference = sorted(reference_peaks, key=lambda peak: peak.energy_eV)
    candidate = sorted(candidate_peaks, key=lambda peak: peak.energy_eV)
    costs = _pair_costs(
        reference,
        candidate,
        maximum_energy_difference_eV,
        float(energy_cost_scale_eV),
        weight_cost_factor,
        use_fractional_weights,
    )
    assignments = (
        _monotone_assignment(costs, unmatched_penalty)
        if method == "monotone"
        else _hungarian_assignment(costs, unmatched_penalty)
    )

    rows: List[SidebandMatchRow] = []
    for status, ref_index, candidate_index, cost in assignments:
        ref_peak = reference[ref_index] if ref_index >= 0 else None
        candidate_peak = (
            candidate[candidate_index] if candidate_index >= 0 else None
        )
        reference_energy = (
            ref_peak.energy_eV if ref_peak is not None else float("nan")
        )
        candidate_energy = (
            candidate_peak.energy_eV
            if candidate_peak is not None
            else float("nan")
        )
        reference_weight = (
            _peak_weight(ref_peak, use_fractional_weights)
            if ref_peak is not None
            else 0.0
        )
        candidate_weight = (
            _peak_weight(candidate_peak, use_fractional_weights)
            if candidate_peak is not None
            else 0.0
        )
        energy_shift = (
            candidate_energy - reference_energy
            if status == "matched"
            else float("nan")
        )
        weight_difference = candidate_weight - reference_weight
        rows.append(
            SidebandMatchRow(
                status=status,
                reference_ordinal=(
                    ref_peak.ordinal if ref_peak is not None else 0
                ),
                candidate_ordinal=(
                    candidate_peak.ordinal if candidate_peak is not None else 0
                ),
                reference_energy_eV=float(reference_energy),
                candidate_energy_eV=float(candidate_energy),
                energy_shift_candidate_minus_reference_eV=float(energy_shift),
                reference_weight=float(reference_weight),
                candidate_weight=float(candidate_weight),
                weight_difference_candidate_minus_reference=float(
                    weight_difference
                ),
                absolute_weight_error=float(abs(weight_difference)),
                assignment_cost=float(cost),
            )
        )

    rows.sort(
        key=lambda row: (
            min(
                row.reference_energy_eV
                if np.isfinite(row.reference_energy_eV)
                else np.inf,
                row.candidate_energy_eV
                if np.isfinite(row.candidate_energy_eV)
                else np.inf,
            ),
            row.status,
        )
    )
    matched_shifts = np.asarray(
        [
            row.energy_shift_candidate_minus_reference_eV
            for row in rows
            if row.status == "matched"
        ],
        dtype=float,
    )
    if matched_shifts.size:
        energy_mae = float(np.mean(np.abs(matched_shifts)))
        energy_rmse = float(np.sqrt(np.mean(matched_shifts**2)))
        energy_max = float(np.max(np.abs(matched_shifts)))
    else:
        energy_mae = energy_rmse = energy_max = float("nan")
    reference_total = float(
        sum(_peak_weight(peak, use_fractional_weights) for peak in reference)
    )
    candidate_total = float(
        sum(_peak_weight(peak, use_fractional_weights) for peak in candidate)
    )
    absolute_weight_error = float(
        sum(row.absolute_weight_error for row in rows)
    )
    relative_weight_error = (
        absolute_weight_error / reference_total
        if reference_total > 0.0
        else float("nan")
    )
    return SidebandMatchResult(
        method=method,
        maximum_energy_difference_eV=float(maximum_energy_difference_eV),
        energy_cost_scale_eV=float(energy_cost_scale_eV),
        weight_cost_factor=float(weight_cost_factor),
        unmatched_penalty=float(unmatched_penalty),
        matched_count=sum(row.status == "matched" for row in rows),
        missing_count=sum(row.status == "missing" for row in rows),
        spurious_count=sum(row.status == "spurious" for row in rows),
        matched_energy_mae_eV=energy_mae,
        matched_energy_rmse_eV=energy_rmse,
        matched_energy_max_abs_eV=energy_max,
        total_reference_weight=reference_total,
        total_candidate_weight=candidate_total,
        aggregate_absolute_weight_error=absolute_weight_error,
        aggregate_relative_weight_error=float(relative_weight_error),
        signed_total_weight_error_candidate_minus_reference=(
            candidate_total - reference_total
        ),
        rows=tuple(rows),
    )


def dark_window_weight(
    grid: Sequence[float],
    spectrum: Sequence[float],
    dark_energy_eV: float,
    half_width_eV: float,
    analysis_window: Optional[Tuple[float, float]] = None,
) -> WindowWeight:
    """Integrate spectral weight around the declared bare dark energy."""

    if not np.isfinite(dark_energy_eV):
        raise ValueError("dark_energy_eV must be finite")
    if not np.isfinite(half_width_eV) or half_width_eV <= 0.0:
        raise ValueError("half_width_eV must be finite and positive")
    energy, _ = validate_uniform_grid(grid)
    values = _validate_spectrum("spectrum", spectrum, energy.shape)
    analysis_lower, analysis_upper = _resolve_window(energy, analysis_window)
    lower = dark_energy_eV - half_width_eV
    upper = dark_energy_eV + half_width_eV
    if lower < analysis_lower or upper > analysis_upper:
        raise ValueError("Dark-energy window lies outside the analysis window")
    total = raw_area(energy, values, (analysis_lower, analysis_upper))
    target = raw_area(energy, values, (lower, upper))
    fraction = target / total if total != 0.0 else float("nan")
    return WindowWeight(
        center_eV=float(dark_energy_eV),
        half_width_eV=float(half_width_eV),
        lower_eV=float(lower),
        upper_eV=float(upper),
        raw_weight=target,
        analysis_raw_area=total,
        fraction_of_analysis_area=float(fraction),
    )


def exact_projected_self_energy(
    grid: Sequence[float],
    projected_green: Sequence[complex],
    bare_projected_energy_eV: float,
    eta_eV: float,
    minimum_green_magnitude: float = 1.0e-14,
) -> np.ndarray:
    r"""Return the operational projected ED self-energy on the whole grid.

    It is defined by the scalar Dyson identity

    ``Sigma_proj(w) = w + i*eta - E_bare - 1/G_proj(w)``.

    This is a projection diagnostic, not a claim that the exact dimer
    self-energy is site-local.  Points where the projected Green function is
    numerically zero are returned as complex NaN.
    """

    energy, _ = validate_uniform_grid(grid)
    green = np.asarray(projected_green, dtype=complex)
    if green.ndim != 1 or green.shape != energy.shape:
        raise ValueError("projected_green and grid shapes differ")
    if not np.all(np.isfinite(green.real)) or not np.all(np.isfinite(green.imag)):
        raise ValueError("projected_green contains NaN or Inf")
    if not np.isfinite(bare_projected_energy_eV):
        raise ValueError("bare_projected_energy_eV must be finite")
    if not np.isfinite(eta_eV) or eta_eV <= 0.0:
        raise ValueError("eta_eV must be finite and positive")
    if minimum_green_magnitude <= 0.0:
        raise ValueError("minimum_green_magnitude must be positive")
    result = np.full(green.shape, np.nan + 1.0j * np.nan, dtype=complex)
    valid = np.abs(green) >= minimum_green_magnitude
    result[valid] = (
        energy[valid]
        + 1.0j * eta_eV
        - bare_projected_energy_eV
        - 1.0 / green[valid]
    )
    return result


def projected_self_energy_at_peak(
    grid: Sequence[float],
    projected_green: Sequence[complex],
    bare_projected_energy_eV: float,
    eta_eV: float,
    peak_energy_eV: float,
    minimum_green_magnitude: float = 1.0e-14,
) -> ProjectedSelfEnergyAtPeak:
    """Evaluate the complex projected Dyson self-energy at a peak energy."""

    energy, _ = validate_uniform_grid(grid)
    green = np.asarray(projected_green, dtype=complex)
    if green.ndim != 1 or green.shape != energy.shape:
        raise ValueError("projected_green and grid shapes differ")
    if not np.all(np.isfinite(green.real)) or not np.all(np.isfinite(green.imag)):
        raise ValueError("projected_green contains NaN or Inf")
    if peak_energy_eV < energy[0] or peak_energy_eV > energy[-1]:
        raise ValueError("peak_energy_eV lies outside the common grid")
    if not np.isfinite(bare_projected_energy_eV):
        raise ValueError("bare_projected_energy_eV must be finite")
    if not np.isfinite(eta_eV) or eta_eV <= 0.0:
        raise ValueError("eta_eV must be finite and positive")
    green_at_peak = np.interp(
        peak_energy_eV, energy, green.real
    ) + 1.0j * np.interp(peak_energy_eV, energy, green.imag)
    if abs(green_at_peak) < minimum_green_magnitude:
        raise ValueError("Projected Green function vanishes at the peak energy")
    sigma = (
        peak_energy_eV
        + 1.0j * eta_eV
        - bare_projected_energy_eV
        - 1.0 / green_at_peak
    )
    return ProjectedSelfEnergyAtPeak(
        energy_eV=float(peak_energy_eV),
        bare_projected_energy_eV=float(bare_projected_energy_eV),
        eta_eV=float(eta_eV),
        green_real_per_eV=float(green_at_peak.real),
        green_imag_per_eV=float(green_at_peak.imag),
        sigma_real_eV=float(sigma.real),
        sigma_imag_eV=float(sigma.imag),
        minus_two_imag_sigma_eV=float(-2.0 * sigma.imag),
    )


def flat_dict(value: object, prefix: str = "") -> Dict[str, object]:
    """Flatten nested metric dataclasses into CSV-friendly scalar columns."""

    if not hasattr(value, "__dataclass_fields__"):
        raise TypeError("flat_dict expects a metric dataclass instance")

    result: Dict[str, object] = {}

    def visit(item: object, stem: str) -> None:
        if hasattr(item, "__dataclass_fields__"):
            for key, nested in asdict(item).items():
                visit(nested, stem + key + "_")
        elif isinstance(item, dict):
            for key, nested in item.items():
                visit(nested, stem + str(key) + "_")
        elif isinstance(item, (list, tuple)):
            # Row collections belong in their own machine-readable tables.
            result[stem[:-1]] = item
        else:
            result[stem[:-1]] = item

    visit(value, prefix)
    return result


__all__ = [
    "PeakAnalysis",
    "PeakComparison",
    "PeakRecord",
    "ProjectedSelfEnergyAtPeak",
    "SidebandMatchResult",
    "SidebandMatchRow",
    "SpectralComparison",
    "WindowMoments",
    "WindowWeight",
    "analyze_peaks",
    "compare_peak_analyses",
    "compare_spectra",
    "dark_window_weight",
    "exact_projected_self_energy",
    "finite_window_moments",
    "flat_dict",
    "match_sidebands",
    "projected_self_energy_at_peak",
    "raw_area",
    "validate_uniform_grid",
]
