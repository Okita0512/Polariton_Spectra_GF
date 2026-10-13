"""Validated access to the finite-temperature Drude dimer spectra.

Large HEOM and SCBA arrays live once under ``Figures/common/dimer/drude/raw``.  Figure
workflows select them with a small project-relative manifest and use the
helpers here for metadata checks, common-support restriction, and area
normalization.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np


EV_PER_AU = 27.211397
EXPECTED_J_MEV = (-100.0, -50.0, -25.0, -12.4, 0.0, 12.4, 25.0, 50.0, 100.0)
EXPECTED_LAMBDA_MEV = 30.0
EXPECTED_OMEGA_F_MEV = 24.8
EXPECTED_TEMPERATURE_K = 300.0
EXPECTED_VERTICAL_ENERGY_EV = 2.030
GRID_MATCH_TOL_EV = 2.0e-12
AREA_TOL = 5.0e-12
HEOM_RINGING_RELATIVE_TOL = 1.0e-4


@dataclass(frozen=True)
class SourceSpec:
    """Project-relative source paths for one signed-J case."""

    j_mev: float
    heom_case_dir: str
    heom_energy_file: str
    heom_response_file: str
    heom_metadata_file: str
    scba_spectrum_file: str


@dataclass(frozen=True)
class SourceCase:
    """Native-grid source arrays for one signed-J case."""

    spec: SourceSpec
    heom_energy_eV: np.ndarray
    heom_absorption_per_eV: np.ndarray
    scba_energy_eV: np.ndarray
    scba_absorption_per_eV: np.ndarray


@dataclass(frozen=True)
class NormalizedCase:
    """Common-support spectra and their normalization diagnostics."""

    source: SourceCase
    energy_eV: np.ndarray
    heom_normalized_per_eV: np.ndarray
    scba_normalized_per_eV: np.ndarray
    interpolation_used: bool
    support_min_eV: float
    support_max_eV: float
    heom_raw_area: float
    scba_raw_area: float
    heom_nonnegative_area: float
    scba_nonnegative_area: float
    heom_normalized_area: float
    scba_normalized_area: float
    heom_negative_area: float
    scba_negative_area: float

    @property
    def j_mev(self) -> float:
        return self.source.spec.j_mev


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility wrapper around NumPy's trapezoidal rule."""

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def parse_validation(path: Path) -> dict[str, str]:
    """Read the upstream ``name = value`` validation format."""

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if " = " in line:
            name, value = line.split(" = ", maxsplit=1)
            values[name.strip()] = value.strip()
    return values


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def resolve_project_file(project_root: Path, relative_path: str) -> Path:
    """Resolve a manifest path and reject absolute or out-of-project targets."""

    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise RuntimeError(f"Manifest path must be project relative: {relative_path}")
    root = project_root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Manifest path leaves project root: {relative_path}") from error
    if not resolved.is_file():
        raise RuntimeError(f"Manifest source does not exist: {relative_path}")
    return resolved


def load_specs(manifest_path: Path) -> list[SourceSpec]:
    """Load the figure manifest and require the established signed-J scan."""

    rows = load_csv(manifest_path)
    required = {
        "J_meV",
        "heom_case_dir",
        "heom_energy_file",
        "heom_response_file",
        "heom_metadata_file",
        "scba_spectrum_file",
    }
    missing = required.difference(rows[0])
    if missing:
        raise RuntimeError(f"Manifest is missing columns: {sorted(missing)}")
    specs = [
        SourceSpec(
            j_mev=float(row["J_meV"]),
            heom_case_dir=row["heom_case_dir"],
            heom_energy_file=row["heom_energy_file"],
            heom_response_file=row["heom_response_file"],
            heom_metadata_file=row["heom_metadata_file"],
            scba_spectrum_file=row["scba_spectrum_file"],
        )
        for row in rows
    ]
    specs.sort(key=lambda spec: spec.j_mev)
    actual_j = tuple(spec.j_mev for spec in specs)
    if actual_j != EXPECTED_J_MEV:
        raise RuntimeError(f"Signed-J scan is {actual_j}, expected {EXPECTED_J_MEV}")
    return specs


def validate_upstream(project_root: Path, specs: list[SourceSpec]) -> None:
    """Require the established HEOM and SCBA source-workflow checks."""

    source_root = project_root / "Figures" / "common" / "dimer" / "drude" / "raw"
    heom_root = source_root / "heom"
    scba_root = source_root / "scba"

    scba_validation = parse_validation(scba_root / "dimer_scba_validation.txt")
    if scba_validation.get("status") != "passed":
        raise RuntimeError("Upstream SCBA validation did not pass")
    for name, expected in {
        "lambda_meV_per_site": EXPECTED_LAMBDA_MEV,
        "omega_f_meV": EXPECTED_OMEGA_F_MEV,
        "temperature_K": EXPECTED_TEMPERATURE_K,
    }.items():
        actual = float(scba_validation.get(name, "nan"))
        if not np.isclose(actual, expected, rtol=0.0, atol=1.0e-12):
            raise RuntimeError(f"SCBA {name}={actual}; expected {expected}")
    if float(scba_validation["maximum_positive_Im_Sigma_au"]) >= 0.0:
        raise RuntimeError("SCBA retarded-causality validation failed")
    if float(scba_validation["minimum_spectral_eigenvalue_au_inverse"]) <= 0.0:
        raise RuntimeError("SCBA spectrum has a negative spectral eigenvalue")

    heom_validation = parse_validation(heom_root / "heom_run_validation.txt")
    if heom_validation.get("status") != "passed":
        raise RuntimeError("Upstream HEOM returned-run validation did not pass")

    heom_manifest = {
        round(float(row["J_meV"]), 9): row
        for row in load_csv(heom_root / "manifest.csv")
    }
    scba_summary = {
        round(float(row["J_meV"]), 9): row
        for row in load_csv(scba_root / "dimer_scba_J_scan_summary.csv")
    }
    for spec in specs:
        key = round(spec.j_mev, 9)
        if key not in heom_manifest or key not in scba_summary:
            raise RuntimeError(f"J={spec.j_mev:g}: absent from an upstream manifest")
        if heom_manifest[key]["case_dir"] != spec.heom_case_dir:
            raise RuntimeError(f"J={spec.j_mev:g}: HEOM case directory mismatch")
        scba_row = scba_summary[key]
        if Path(spec.scba_spectrum_file).name != scba_row["spectrum_file"]:
            raise RuntimeError(f"J={spec.j_mev:g}: SCBA spectrum path mismatch")
        if abs(float(scba_row["bright_sum_rule"]) - 1.0) > 1.0e-3:
            raise RuntimeError(f"J={spec.j_mev:g}: bright sum rule failed")
        for name, expected in (
            ("lambda_meV", EXPECTED_LAMBDA_MEV),
            ("omega_f_meV", EXPECTED_OMEGA_F_MEV),
            ("vertical_site_energy_eV", EXPECTED_VERTICAL_ENERGY_EV),
        ):
            actual = float(scba_row[name])
            if not np.isclose(actual, expected, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(
                    f"SCBA J={spec.j_mev:g}: {name}={actual}; expected {expected}"
                )


def load_sources(project_root: Path, manifest_path: Path) -> list[SourceCase]:
    """Load the figure-selected raw arrays after validating their provenance."""

    specs = load_specs(manifest_path)
    validate_upstream(project_root, specs)
    cases: list[SourceCase] = []

    for spec in specs:
        energy_path = resolve_project_file(project_root, spec.heom_energy_file)
        response_path = resolve_project_file(project_root, spec.heom_response_file)
        metadata_path = resolve_project_file(project_root, spec.heom_metadata_file)
        scba_path = resolve_project_file(project_root, spec.scba_spectrum_file)

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for name, expected in {
            "J_meV": spec.j_mev,
            "lambda_meV_per_site": EXPECTED_LAMBDA_MEV,
            "omega_f_meV": EXPECTED_OMEGA_F_MEV,
            "temperature_K": EXPECTED_TEMPERATURE_K,
            "vertical_site_energy_eV": EXPECTED_VERTICAL_ENERGY_EV,
        }.items():
            actual = float(metadata.get(name, np.nan))
            if not np.isclose(actual, expected, rtol=0.0, atol=1.0e-12):
                raise RuntimeError(
                    f"{spec.heom_case_dir}: {name}={actual}; expected {expected}"
                )

        heom_energy = np.asarray(np.loadtxt(energy_path), dtype=float).reshape(-1)
        heom_energy *= EV_PER_AU
        heom_spectrum = np.asarray(np.loadtxt(response_path), dtype=float).reshape(-1)
        heom_spectrum /= EV_PER_AU
        scba_values = np.asarray(np.loadtxt(scba_path), dtype=float)
        if scba_values.ndim != 2 or scba_values.shape[1] < 2:
            raise RuntimeError(f"Malformed SCBA spectrum: {spec.scba_spectrum_file}")
        scba_energy = scba_values[:, 0]
        scba_spectrum = scba_values[:, 1]

        arrays = (heom_energy, heom_spectrum, scba_energy, scba_spectrum)
        if any(array.size < 2 for array in arrays):
            raise RuntimeError(f"J={spec.j_mev:g}: source array is too short")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise RuntimeError(f"J={spec.j_mev:g}: non-finite source value")
        if np.any(np.diff(heom_energy) <= 0.0) or np.any(np.diff(scba_energy) <= 0.0):
            raise RuntimeError(f"J={spec.j_mev:g}: non-monotone energy grid")
        if float(np.min(scba_spectrum)) < -1.0e-14:
            raise RuntimeError(f"J={spec.j_mev:g}: negative SCBA spectral density")
        heom_floor_relative = max(
            0.0, -float(np.min(heom_spectrum)) / float(np.max(heom_spectrum))
        )
        if heom_floor_relative > HEOM_RINGING_RELATIVE_TOL:
            raise RuntimeError(
                f"J={spec.j_mev:g}: HEOM negative Fourier ringing is too large "
                f"({heom_floor_relative:.3e})"
            )

        cases.append(
            SourceCase(
                spec=spec,
                heom_energy_eV=heom_energy,
                heom_absorption_per_eV=heom_spectrum,
                scba_energy_eV=scba_energy,
                scba_absorption_per_eV=scba_spectrum,
            )
        )
    return cases


def restrict_to_support(
    energy: np.ndarray,
    spectrum: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Restrict to closed support, interpolating an endpoint only if absent."""

    mask = (energy >= lower - GRID_MATCH_TOL_EV) & (
        energy <= upper + GRID_MATCH_TOL_EV
    )
    selected_energy = energy[mask].copy()
    selected_spectrum = spectrum[mask].copy()
    if selected_energy.size < 2:
        raise RuntimeError("Too few points on common physical support")

    interpolation_used = False
    if abs(float(selected_energy[0]) - lower) <= GRID_MATCH_TOL_EV:
        selected_energy[0] = lower
    else:
        selected_energy = np.insert(selected_energy, 0, lower)
        selected_spectrum = np.insert(
            selected_spectrum, 0, np.interp(lower, energy, spectrum)
        )
        interpolation_used = True
    if abs(float(selected_energy[-1]) - upper) <= GRID_MATCH_TOL_EV:
        selected_energy[-1] = upper
    else:
        selected_energy = np.append(selected_energy, upper)
        selected_spectrum = np.append(
            selected_spectrum, np.interp(upper, energy, spectrum)
        )
        interpolation_used = True
    return selected_energy, selected_spectrum, interpolation_used


def common_support_arrays(
    case: SourceCase,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, float, float]:
    """Return both spectra on one grid without extrapolated spectral mass."""

    lower = max(float(case.heom_energy_eV[0]), float(case.scba_energy_eV[0]))
    upper = min(float(case.heom_energy_eV[-1]), float(case.scba_energy_eV[-1]))
    if not lower < upper:
        raise RuntimeError(f"J={case.spec.j_mev:g}: source supports do not overlap")

    he, ha, heom_boundary_interp = restrict_to_support(
        case.heom_energy_eV, case.heom_absorption_per_eV, lower, upper
    )
    se, sa, scba_boundary_interp = restrict_to_support(
        case.scba_energy_eV, case.scba_absorption_per_eV, lower, upper
    )
    if len(he) == len(se) and np.allclose(
        he, se, rtol=0.0, atol=GRID_MATCH_TOL_EV
    ):
        return he, ha, sa, heom_boundary_interp or scba_boundary_interp, lower, upper

    common_energy = np.unique(np.concatenate((he, se)))
    common_energy = common_energy[
        (common_energy >= lower) & (common_energy <= upper)
    ]
    if common_energy[0] != lower:
        common_energy = np.insert(common_energy, 0, lower)
    if common_energy[-1] != upper:
        common_energy = np.append(common_energy, upper)
    heom_common = np.interp(common_energy, he, ha)
    scba_common = np.interp(common_energy, se, sa)
    return common_energy, heom_common, scba_common, True, lower, upper


def normalize_case(case: SourceCase) -> NormalizedCase:
    """Area-normalize one pair on its common support.

    The raw HEOM Fourier transform contains a tiny negative far-tail ringing
    floor.  Its signed area and removed mass are retained as diagnostics, while
    the nonnegative density used in the figure is obtained by clipping only
    negative samples to zero before unit-area normalization.
    """

    energy, heom_raw, scba_raw, interpolated, lower, upper = common_support_arrays(case)
    heom_raw_area = trapezoid(heom_raw, energy)
    scba_raw_area = trapezoid(scba_raw, energy)
    if heom_raw_area <= 0.0 or scba_raw_area <= 0.0:
        raise RuntimeError(f"J={case.spec.j_mev:g}: nonpositive raw area")

    heom_nonnegative = np.maximum(heom_raw, 0.0)
    scba_nonnegative = np.maximum(scba_raw, 0.0)
    heom_nonnegative_area = trapezoid(heom_nonnegative, energy)
    scba_nonnegative_area = trapezoid(scba_nonnegative, energy)
    if heom_nonnegative_area <= 0.0 or scba_nonnegative_area <= 0.0:
        raise RuntimeError(f"J={case.spec.j_mev:g}: nonpositive clipped area")

    heom_normalized = heom_nonnegative / heom_nonnegative_area
    scba_normalized = scba_nonnegative / scba_nonnegative_area
    heom_normalized_area = trapezoid(heom_normalized, energy)
    scba_normalized_area = trapezoid(scba_normalized, energy)
    if abs(heom_normalized_area - 1.0) > AREA_TOL:
        raise RuntimeError(f"J={case.spec.j_mev:g}: HEOM normalization failed")
    if abs(scba_normalized_area - 1.0) > AREA_TOL:
        raise RuntimeError(f"J={case.spec.j_mev:g}: SCBA normalization failed")

    return NormalizedCase(
        source=case,
        energy_eV=energy,
        heom_normalized_per_eV=heom_normalized,
        scba_normalized_per_eV=scba_normalized,
        interpolation_used=interpolated,
        support_min_eV=lower,
        support_max_eV=upper,
        heom_raw_area=heom_raw_area,
        scba_raw_area=scba_raw_area,
        heom_nonnegative_area=heom_nonnegative_area,
        scba_nonnegative_area=scba_nonnegative_area,
        heom_normalized_area=heom_normalized_area,
        scba_normalized_area=scba_normalized_area,
        heom_negative_area=max(
            0.0, -trapezoid(np.minimum(heom_raw, 0.0), energy)
        ),
        scba_negative_area=max(
            0.0, -trapezoid(np.minimum(scba_raw, 0.0), energy)
        ),
    )


def write_area_validation_csv(
    path: Path,
    cases: Iterable[NormalizedCase],
    display_window_eV: tuple[float, float],
) -> None:
    """Write figure-specific raw-area and normalization diagnostics."""

    fieldnames = [
        "J_meV",
        "heom_raw_area",
        "scba_raw_area",
        "raw_area_relative_difference_HEOM_over_SCBA_minus_1",
        "heom_nonnegative_area_used_for_normalization",
        "scba_nonnegative_area_used_for_normalization",
        "heom_clipped_negative_area",
        "scba_clipped_negative_area",
        "heom_negative_area_fraction",
        "scba_negative_area_fraction",
        "heom_normalized_area",
        "scba_normalized_area",
        "heom_normalization_error",
        "scba_normalization_error",
        "support_min_eV",
        "support_max_eV",
        "common_grid_points",
        "common_grid_spacing_meV",
        "interpolation_used",
        "heom_min_normalized_density_in_display_window_per_eV",
        "scba_min_normalized_density_in_display_window_per_eV",
        "normalization_status",
        "heom_energy_file",
        "heom_response_file",
        "scba_spectrum_file",
    ]
    rows: list[dict[str, object]] = []
    for case in cases:
        display = (case.energy_eV >= display_window_eV[0]) & (
            case.energy_eV <= display_window_eV[1]
        )
        rows.append(
            {
                "J_meV": f"{case.j_mev:g}",
                "heom_raw_area": f"{case.heom_raw_area:.15f}",
                "scba_raw_area": f"{case.scba_raw_area:.15f}",
                "raw_area_relative_difference_HEOM_over_SCBA_minus_1": (
                    f"{case.heom_raw_area / case.scba_raw_area - 1.0:.15e}"
                ),
                "heom_nonnegative_area_used_for_normalization": (
                    f"{case.heom_nonnegative_area:.15f}"
                ),
                "scba_nonnegative_area_used_for_normalization": (
                    f"{case.scba_nonnegative_area:.15f}"
                ),
                "heom_clipped_negative_area": f"{case.heom_negative_area:.15e}",
                "scba_clipped_negative_area": f"{case.scba_negative_area:.15e}",
                "heom_negative_area_fraction": (
                    f"{case.heom_negative_area / case.heom_raw_area:.15e}"
                ),
                "scba_negative_area_fraction": (
                    f"{case.scba_negative_area / case.scba_raw_area:.15e}"
                ),
                "heom_normalized_area": f"{case.heom_normalized_area:.15f}",
                "scba_normalized_area": f"{case.scba_normalized_area:.15f}",
                "heom_normalization_error": (
                    f"{case.heom_normalized_area - 1.0:.15e}"
                ),
                "scba_normalization_error": (
                    f"{case.scba_normalized_area - 1.0:.15e}"
                ),
                "support_min_eV": f"{case.support_min_eV:.9f}",
                "support_max_eV": f"{case.support_max_eV:.9f}",
                "common_grid_points": len(case.energy_eV),
                "common_grid_spacing_meV": (
                    f"{1.0e3 * float(np.median(np.diff(case.energy_eV))):.9f}"
                ),
                "interpolation_used": str(case.interpolation_used).lower(),
                "heom_min_normalized_density_in_display_window_per_eV": (
                    f"{float(np.min(case.heom_normalized_per_eV[display])):.15e}"
                ),
                "scba_min_normalized_density_in_display_window_per_eV": (
                    f"{float(np.min(case.scba_normalized_per_eV[display])):.15e}"
                ),
                "normalization_status": "passed",
                "heom_energy_file": case.source.spec.heom_energy_file,
                "heom_response_file": case.source.spec.heom_response_file,
                "scba_spectrum_file": case.source.spec.scba_spectrum_file,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_area_validation_report(
    path: Path,
    cases: list[NormalizedCase],
    display_window_eV: tuple[float, float],
) -> None:
    """Write an aggregate validation report for the nine-panel montage."""

    supports = {
        (round(case.support_min_eV, 12), round(case.support_max_eV, 12))
        for case in cases
    }
    if len(supports) != 1:
        raise RuntimeError(f"Cases do not share one common support: {supports}")
    lower, upper = next(iter(supports))
    max_normalization_error = max(
        max(
            abs(case.heom_normalized_area - 1.0),
            abs(case.scba_normalized_area - 1.0),
        )
        for case in cases
    )
    min_display_density = min(
        min(
            float(np.min(case.heom_normalized_per_eV[
                (case.energy_eV >= display_window_eV[0])
                & (case.energy_eV <= display_window_eV[1])
            ])),
            float(np.min(case.scba_normalized_per_eV[
                (case.energy_eV >= display_window_eV[0])
                & (case.energy_eV <= display_window_eV[1])
            ])),
        )
        for case in cases
    )
    lines = [
        "Fig. S9 Drude--Lorentz dimer area-normalization validation",
        "status = passed",
        f"spectral_panels = {len(cases)}",
        "panel_j_present = false",
        "W1_computed_by_FigS9_driver = false",
        f"signed_J_meV = {', '.join(f'{case.j_mev:g}' for case in cases)}",
        f"common_support_eV = {lower:.4f} to {upper:.4f}",
        f"display_window_eV = {display_window_eV[0]:.2f} to {display_window_eV[1]:.2f}",
        "integration = trapezoidal area integral",
        "normalization = each nonnegative response separately normalized on common support",
        "negative_HEOM_tail_handling = tiny Fourier-ringing samples clipped to zero before montage normalization; raw signed area and removed mass retained in CSV",
        f"interpolated_cases = {sum(case.interpolation_used for case in cases)}",
        f"maximum_normalization_error = {max_normalization_error:.3e}",
        "maximum_HEOM_negative_area_fraction = "
        f"{max(case.heom_negative_area / case.heom_raw_area for case in cases):.3e}",
        "maximum_SCBA_negative_area_fraction = "
        f"{max(case.scba_negative_area / case.scba_raw_area for case in cases):.3e}",
        f"minimum_display_window_normalized_density_per_eV = {min_display_density:.3e}",
        "upstream_SCBA_validation = passed (causality, positivity, sum rule, convergence)",
        "upstream_HEOM_returned_run_validation = passed",
        "physical_parameters_retuned = false",
        "omega_f_role = overdamped Drude bath correlation/cutoff scale",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
