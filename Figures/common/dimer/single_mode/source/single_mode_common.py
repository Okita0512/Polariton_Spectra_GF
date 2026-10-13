#!/usr/bin/env python3
"""Shared model, grid, naming, and serialization utilities.

All energies used by the numerical solvers are in eV and hbar=1.  This module
has no dependency on the process working directory: paths are resolved from
the source file location.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np


KB_EV_PER_K = 8.617333262145e-5
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SPECTRA = RESULTS / "spectra"
FIGURES = ROOT / "figures"
VALIDATION = ROOT / "validation"


@dataclass(frozen=True)
class ModelParams:
    """Physical parameters for the Frenkel dimer and two local modes."""

    energy_00_eV: float = 2.0
    omega0_meV: float = 24.8
    huang_rhys: float = 30.0 / 24.8
    j_meV: float = 0.0
    temperature_K: float = 300.0
    eta_meV: float = 0.5

    @property
    def omega0_eV(self) -> float:
        return self.omega0_meV * 1.0e-3

    @property
    def j_eV(self) -> float:
        return self.j_meV * 1.0e-3

    @property
    def eta_eV(self) -> float:
        return self.eta_meV * 1.0e-3

    @property
    def reorganization_eV(self) -> float:
        return self.huang_rhys * self.omega0_eV

    @property
    def vertical_energy_eV(self) -> float:
        return self.energy_00_eV + self.reorganization_eV

    @property
    def coupling_eV(self) -> float:
        return self.omega0_eV * np.sqrt(self.huang_rhys)

    @property
    def coupling_squared_eV2(self) -> float:
        return self.huang_rhys * self.omega0_eV**2

    @property
    def resonance_ratio(self) -> float:
        return 2.0 * self.j_eV / self.omega0_eV

    @property
    def bose_occupation(self) -> float:
        if self.temperature_K <= 0.0:
            return 0.0
        x = self.omega0_eV / (KB_EV_PER_K * self.temperature_K)
        return float(1.0 / np.expm1(x))

    def validate(self) -> None:
        values = asdict(self)
        if not all(np.isfinite(float(value)) for value in values.values()):
            raise ValueError("All physical parameters must be finite")
        if self.omega0_meV <= 0.0:
            raise ValueError("omega0_meV must be positive")
        if self.huang_rhys < 0.0:
            raise ValueError("huang_rhys must be nonnegative")
        if self.temperature_K < 0.0:
            raise ValueError("temperature_K must be nonnegative")
        if self.eta_meV <= 0.0:
            raise ValueError("eta_meV must be positive")


@dataclass(frozen=True)
class GridSpec:
    """Uniform electronic grid and the finite common analysis window."""

    analysis_min_eV: float
    analysis_max_eV: float
    internal_min_eV: float
    internal_max_eV: float
    step_meV: float

    @property
    def step_eV(self) -> float:
        return self.step_meV * 1.0e-3

    def grid(self) -> np.ndarray:
        if not self.internal_min_eV < self.internal_max_eV:
            raise ValueError("Internal grid bounds are unordered")
        if self.step_meV <= 0.0 or not np.isfinite(self.step_meV):
            raise ValueError("Grid spacing must be finite and positive")
        intervals = int(
            np.ceil((self.internal_max_eV - self.internal_min_eV) / self.step_eV)
        )
        # An odd number of samples gives the Lorentz kernel an exact center.
        if (intervals + 1) % 2 == 0:
            intervals += 1
        return self.internal_min_eV + self.step_eV * np.arange(intervals + 1)

    def analysis_mask(self, grid: np.ndarray) -> np.ndarray:
        tolerance = 32.0 * np.finfo(float).eps * max(
            1.0, abs(self.analysis_min_eV), abs(self.analysis_max_eV)
        )
        return (grid >= self.analysis_min_eV - tolerance) & (
            grid <= self.analysis_max_eV + tolerance
        )


def default_grid_spec(
    model: ModelParams,
    step_meV: float = 0.0,
    padding_eV: float = 0.55,
) -> GridSpec:
    """Return a conservative grid for resolved vibronic spectra.

    The declared analysis window is intentionally finite because Lorentzian
    first and higher moments are not defined on the infinite line.  The Dyson
    iteration uses an additional padding region on both sides.
    """

    model.validate()
    if step_meV <= 0.0:
        step_meV = min(
            model.eta_meV / 5.0,
            model.omega0_meV / 200.0,
            0.20,
        )
    omega = model.omega0_eV
    hopping = abs(model.j_eV)
    lower_span = max(0.80, hopping + 4.0 * omega)
    upper_span = hopping + max(
        0.80,
        (model.huang_rhys + 8.0) * omega,
    )
    analysis_min = model.energy_00_eV - lower_span
    analysis_max = model.energy_00_eV + upper_span
    padding = max(padding_eV, 4.0 * omega)
    return GridSpec(
        analysis_min_eV=analysis_min,
        analysis_max_eV=analysis_max,
        internal_min_eV=analysis_min - padding,
        internal_max_eV=analysis_max + padding,
        step_meV=step_meV,
    )


def thermal_cutoff(
    model: ModelParams,
    tolerance: float = 1.0e-10,
) -> Tuple[int, float]:
    """Return states per mode and omitted joint two-mode probability."""

    if not 0.0 < tolerance < 1.0:
        raise ValueError("thermal tolerance must lie between zero and one")
    if model.temperature_K <= 0.0:
        return 1, 0.0
    q = float(
        np.exp(-model.omega0_eV / (KB_EV_PER_K * model.temperature_K))
    )
    n_states = 1
    while True:
        omitted = 1.0 - (1.0 - q**n_states) ** 2
        if omitted < tolerance:
            return n_states, float(omitted)
        n_states += 1
        if n_states > 10000:
            raise RuntimeError("Thermal cutoff search failed")


def recommended_n_vib(model: ModelParams, thermal_states: int) -> int:
    """Conservative factorized ED cutoff used by production scans."""

    displacement_margin = int(
        np.ceil(model.huang_rhys + 8.0 * np.sqrt(model.huang_rhys + 1.0))
    )
    return max(14, thermal_states + displacement_margin)


def signed_tag(value: float, digits: int = 8) -> str:
    """Filesystem-safe, collision-resistant signed decimal tag."""

    if not np.isfinite(value):
        raise ValueError("Cannot tag a non-finite number")
    text = f"{abs(value):.{digits}g}".replace(".", "p").replace("-", "m")
    sign = "m" if value < 0.0 else "p"
    return sign + text


def ensure_directories() -> None:
    for path in (RESULTS, SPECTRA, FIGURES, VALIDATION):
        path.mkdir(parents=True, exist_ok=True)


def encode_metadata(values: Dict[str, Any]) -> np.ndarray:
    return np.asarray(json.dumps(values, sort_keys=True))


def decode_metadata(value: np.ndarray) -> Dict[str, Any]:
    return json.loads(str(np.asarray(value).item()))


def model_dict(model: ModelParams) -> Dict[str, float]:
    result = {key: float(value) for key, value in asdict(model).items()}
    result.update(
        {
            "omega0_eV": model.omega0_eV,
            "j_eV": model.j_eV,
            "eta_eV": model.eta_eV,
            "lambda_eV": model.reorganization_eV,
            "epsilon_v_eV": model.vertical_energy_eV,
            "g_eV": model.coupling_eV,
            "r_2J_over_omega0": model.resonance_ratio,
            "bose_n0": model.bose_occupation,
        }
    )
    return result


def trapezoid(values: np.ndarray, grid: np.ndarray, mask: np.ndarray) -> float:
    return float(np.trapz(values[mask], grid[mask]))


def write_json(path: Path, values: Dict[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_or_raise(name: str, arrays: Iterable[np.ndarray]) -> None:
    for array in arrays:
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"{name} contains NaN or Inf")
