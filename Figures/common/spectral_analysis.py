#!/usr/bin/env python3
"""Shared peak, linewidth, and error analysis for manuscript spectra.

The linewidth reported by this project is a direct full width at half maximum
(FWHM), not a Lorentzian fit parameter.  A lightly smoothed copy of a
normalized spectrum is used only to locate the two dominant polariton peaks.
All baselines and half-height crossings are evaluated on the original data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks, peak_prominences, savgol_filter


@dataclass(frozen=True)
class BranchMetrics:
    center: float
    fwhm: float
    left_half_width: float
    right_half_width: float
    baseline: float
    peak_height: float
    prominence: float
    uncertainty: float
    reliable: bool
    weak: bool


@dataclass(frozen=True)
class TwoPeakMetrics:
    lower: BranchMetrics
    upper: BranchMetrics
    valley: float


def _normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("Spectrum has no finite positive maximum")
    return values / maximum


def _crossing(
    x: np.ndarray,
    y: np.ndarray,
    level: float,
    first: int,
    second: int,
) -> float:
    x0, x1 = float(x[first]), float(x[second])
    y0, y1 = float(y[first]), float(y[second])
    if y1 == y0:
        return x0
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def _branch_metrics(
    x: np.ndarray,
    y: np.ndarray,
    peak: int,
    left_bound: int,
    right_bound: int,
    prominence: float,
    weak_threshold: float,
) -> BranchMetrics:
    segment = y[left_bound : right_bound + 1]
    baseline = float(np.min(segment))
    peak_height = float(y[peak])
    half = baseline + 0.5 * (peak_height - baseline)

    left_crossing = np.nan
    for index in range(peak, left_bound, -1):
        if (y[index - 1] - half) * (y[index] - half) <= 0.0:
            left_crossing = _crossing(x, y, half, index - 1, index)
            break

    right_crossing = np.nan
    for index in range(peak, right_bound):
        if (y[index] - half) * (y[index + 1] - half) <= 0.0:
            right_crossing = _crossing(x, y, half, index, index + 1)
            break

    center = float(x[peak])
    left_half_width = center - left_crossing
    right_half_width = right_crossing - center
    fwhm = right_crossing - left_crossing
    resolution = float(np.median(np.diff(x)))
    reliable = bool(
        np.isfinite(fwhm)
        and fwhm > 2.0 * resolution
        and peak_height > baseline
        and prominence >= 0.005
    )
    return BranchMetrics(
        center=center,
        fwhm=float(fwhm),
        left_half_width=float(left_half_width),
        right_half_width=float(right_half_width),
        baseline=baseline,
        peak_height=peak_height,
        prominence=float(prominence),
        uncertainty=resolution,
        reliable=reliable,
        weak=bool(prominence < weak_threshold),
    )


def analyze_two_peaks(
    x: Sequence[float],
    y: Sequence[float],
    x_range: Optional[Tuple[float, float]] = None,
    prominence: float = 0.01,
    fallback_prominence: float = 0.005,
    minimum_distance: int = 10,
) -> TwoPeakMetrics:
    """Locate two dominant peaks and measure branch-resolved direct FWHM.

    The minimum between the two centers separates the LP and UP domains.
    Within each domain the local minimum is the baseline.  Half-height
    crossings are linearly interpolated on the unsmoothed spectrum.
    """
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 1 or y_array.shape != x_array.shape:
        raise ValueError("x and y must be one-dimensional arrays of equal size")
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    if x_range is not None:
        finite &= (x_array >= x_range[0]) & (x_array <= x_range[1])
    x_array = x_array[finite]
    y_array = y_array[finite]
    if len(x_array) < 21:
        raise ValueError("Too few spectral points for two-peak analysis")
    order = np.argsort(x_array)
    x_array = x_array[order]
    y_array = _normalized(y_array[order])

    window = min(11, len(y_array) if len(y_array) % 2 else len(y_array) - 1)
    smooth = savgol_filter(y_array, window, min(3, window - 1))
    peaks, properties = find_peaks(
        smooth, prominence=prominence, distance=minimum_distance
    )
    if len(peaks) < 2:
        peaks, properties = find_peaks(
            smooth,
            prominence=fallback_prominence,
            distance=max(5, minimum_distance // 2),
        )
    if len(peaks) < 2:
        raise ValueError("Fewer than two resolvable peaks")

    selected = peaks[np.argsort(properties["prominences"])[-2:]]
    lower_peak, upper_peak = sorted(int(index) for index in selected)
    valley = int(
        lower_peak
        + np.argmin(y_array[lower_peak : upper_peak + 1])
    )
    selected_prominences = peak_prominences(
        smooth, np.asarray([lower_peak, upper_peak])
    )[0]
    lower = _branch_metrics(
        x_array,
        y_array,
        lower_peak,
        0,
        valley,
        float(selected_prominences[0]),
        prominence,
    )
    upper = _branch_metrics(
        x_array,
        y_array,
        upper_peak,
        valley,
        len(x_array) - 1,
        float(selected_prominences[1]),
        prominence,
    )
    return TwoPeakMetrics(
        lower=lower,
        upper=upper,
        valley=float(x_array[valley]),
    )


def analyze_single_peak(
    x: Sequence[float],
    y: Sequence[float],
    x_range: Optional[Tuple[float, float]] = None,
) -> BranchMetrics:
    """Measure the dominant peak with the same direct-FWHM protocol."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    if x_range is not None:
        finite &= (x_array >= x_range[0]) & (x_array <= x_range[1])
    x_array = x_array[finite]
    y_array = y_array[finite]
    order = np.argsort(x_array)
    x_array = x_array[order]
    y_array = _normalized(y_array[order])
    peak = int(np.argmax(y_array))
    prominence = float(
        peak_prominences(y_array, np.asarray([peak]))[0][0]
    )
    return _branch_metrics(
        x_array,
        y_array,
        peak,
        0,
        len(x_array) - 1,
        prominence,
        0.01,
    )


def mean_absolute_error(
    calculated: Iterable[float],
    reference: Iterable[float],
) -> float:
    calculated_array = np.asarray(tuple(calculated), dtype=float)
    reference_array = np.asarray(tuple(reference), dtype=float)
    finite = np.isfinite(calculated_array) & np.isfinite(reference_array)
    if not np.any(finite):
        return np.nan
    return float(
        np.mean(np.abs(calculated_array[finite] - reference_array[finite]))
    )


def integrated_normalized_error(
    x: Sequence[float],
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    x_array = np.asarray(x, dtype=float)
    first_array = _normalized(np.asarray(first, dtype=float))
    second_array = _normalized(np.asarray(second, dtype=float))
    numerator = np.trapz(np.abs(first_array - second_array), x_array)
    denominator = np.trapz(np.abs(second_array), x_array)
    return float(numerator / denominator) if denominator > 0.0 else np.nan
