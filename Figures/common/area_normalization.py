#!/usr/bin/env python3
"""Shared area-matching utilities for manuscript spectra."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _finite_sorted(
    energy: Sequence[float],
    values: Sequence[float],
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    energy_array = np.asarray(energy, dtype=float)
    values_array = np.asarray(values, dtype=float)
    if energy_array.shape != values_array.shape or energy_array.ndim != 1:
        raise ValueError("Energy and spectrum must be same-length 1D arrays")
    selected = (
        np.isfinite(energy_array)
        & np.isfinite(values_array)
        & (energy_array >= lower)
        & (energy_array <= upper)
    )
    if np.count_nonzero(selected) < 2:
        raise ValueError("Spectrum has fewer than two finite overlap points")
    order = np.argsort(energy_array[selected])
    return energy_array[selected][order], values_array[selected][order]


def match_area_to_reference(
    reference_energy: Sequence[float],
    reference_values: Sequence[float],
    comparison_energy: Sequence[float],
    comparison_values: Sequence[float],
) -> np.ndarray:
    """Scale one spectrum to the reference area over their common support."""
    reference_energy_array = np.asarray(reference_energy, dtype=float)
    reference_values_array = np.asarray(reference_values, dtype=float)
    comparison_energy_array = np.asarray(comparison_energy, dtype=float)
    comparison_values_array = np.asarray(comparison_values, dtype=float)
    if (
        reference_energy_array.shape != reference_values_array.shape
        or comparison_energy_array.shape != comparison_values_array.shape
    ):
        raise ValueError("Energy and spectrum shapes do not match")
    reference_finite = reference_energy_array[
        np.isfinite(reference_energy_array)
        & np.isfinite(reference_values_array)
    ]
    comparison_finite = comparison_energy_array[
        np.isfinite(comparison_energy_array)
        & np.isfinite(comparison_values_array)
    ]
    if len(reference_finite) < 2 or len(comparison_finite) < 2:
        raise ValueError("Cannot determine common finite spectral support")
    lower = max(
        float(np.min(reference_finite)),
        float(np.min(comparison_finite)),
    )
    upper = min(
        float(np.max(reference_finite)),
        float(np.max(comparison_finite)),
    )
    if lower >= upper:
        raise ValueError("Reference and comparison spectra do not overlap")

    reference_x, reference_y = _finite_sorted(
        reference_energy, reference_values, lower, upper
    )
    comparison_x, comparison_y = _finite_sorted(
        comparison_energy, comparison_values, lower, upper
    )
    reference_area = float(np.trapz(reference_y, reference_x))
    comparison_area = float(np.trapz(comparison_y, comparison_x))
    if (
        not np.isfinite(reference_area)
        or not np.isfinite(comparison_area)
        or reference_area <= 0.0
        or comparison_area <= 0.0
    ):
        raise ValueError(
            "Area matching requires finite positive integrated spectra"
        )
    return np.asarray(comparison_values, dtype=float) * (
        reference_area / comparison_area
    )
