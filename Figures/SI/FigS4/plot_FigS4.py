#!/usr/bin/env python3
"""Plot Fig. S5 with fixed single-molecule coupling and growing splitting."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator
import numpy as np


HUANG_RHYS = (0.5, 1.0, 2.0)
MOLECULE_COUNTS = (1, 2, 4, 8)
ROW_CASES = (None,) + MOLECULE_COUNTS
X_LIMITS = (1.8, 2.3)
Y_LIMITS = (-0.05, 1.95)
PANEL_LABELS = tuple(
    f"({chr(ord('a') + index)})" for index in range(15)
)

FIGURE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_ROOT.parents[2]
PROTOCOL_ROOT = PROJECT_ROOT / "Figures" / "shared" / "discrete_mode" / "zero_zero"
RESULTS_ROOT = PROTOCOL_ROOT / "results"
OUTPUT_ROOT = FIGURE_ROOT
sys.path.insert(0, str(PROJECT_ROOT / "Figures" / "common"))
from area_normalization import match_area_to_reference  # noqa: E402
from si_plot_style import (  # noqa: E402
    EXACT_STYLE,
    SCBA_STYLE,
    add_panel_label,
    compact_legend_kwargs,
    configure_si_matplotlib,
    style_axis as apply_shared_axis_style,
)


def result_path(n_molecules: int | None, s_value: float) -> Path:
    """Return the independently stored result archive for one panel."""

    if n_molecules is None:
        return RESULTS_ROOT / f"outside_S{s_value:.1f}.npz"
    return RESULTS_ROOT / f"N{n_molecules}_S{s_value:.1f}.npz"


def load_case(n_molecules: int | None, s_value: float) -> dict[str, object]:
    """Load one archive and reject stale or mismatched physical parameters."""

    path = result_path(n_molecules, s_value)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run calculate_FigS4.py first"
        )
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        result: dict[str, object] = {
            "path": path,
            "exact_energy": archive["exact_energy_eV"].copy(),
            "exact_spectrum": archive["exact_spectrum_per_eV"].copy(),
            "scba_energy": archive["scba_energy_eV"].copy(),
            "scba_spectrum": archive["scba_spectrum_per_eV"].copy(),
            "metadata": metadata,
        }

    model = metadata["model"]
    numerics = metadata["numerics"]
    expected_model = {
        "energy_00": 2.0,
        "phonon_frequency": 0.02,
        "eta": 0.0005,
        "huang_rhys": s_value,
    }
    for key, value in expected_model.items():
        if abs(float(model[key]) - value) > 1.0e-12:
            raise ValueError(f"{key} metadata mismatch in {path}")

    inside = n_molecules is not None
    if bool(model["inside_cavity"]) != inside:
        raise ValueError(f"Cavity-condition metadata mismatch in {path}")
    if inside:
        if int(model["n_molecules"]) != n_molecules:
            raise ValueError(f"Molecule-count metadata mismatch in {path}")
        if abs(float(model["cavity_frequency"]) - 2.0) > 1.0e-12:
            raise ValueError(f"Cavity-frequency metadata mismatch in {path}")
        if (
            abs(float(model["fixed_single_molecule_coupling"]) - 0.03536)
            > 1.0e-12
        ):
            raise ValueError(f"Fixed-g_c metadata mismatch in {path}")
        if abs(float(model["single_molecule_coupling"]) - 0.03536) > 1.0e-12:
            raise ValueError(f"Single-molecule coupling mismatch in {path}")
        expected_collective = np.sqrt(n_molecules) * 0.03536
        if (
            abs(float(model["collective_coupling"]) - expected_collective)
            > 1.0e-12
        ):
            raise ValueError(f"Collective-coupling metadata mismatch in {path}")
        if abs(
            np.sqrt(n_molecules)
            * float(model["single_molecule_coupling"])
            - expected_collective
        ) > 1.0e-13:
            raise ValueError(f"sqrt(N) g_c metadata mismatch in {path}")
    if int(numerics["ed_total_phonon_cutoff"]) != 28:
        raise ValueError(f"ED cutoff metadata mismatch in {path}")
    if int(numerics["ed_check_total_phonon_cutoff"]) != 30:
        raise ValueError(f"ED check-cutoff metadata mismatch in {path}")
    if int(numerics["ed_lanczos_steps"]) != 5000:
        raise ValueError(f"Lanczos-production metadata mismatch in {path}")
    if int(numerics["ed_lanczos_check_steps"]) != 4000:
        raise ValueError(f"Lanczos-check metadata mismatch in {path}")

    for key in (
        "exact_energy",
        "exact_spectrum",
        "scba_energy",
        "scba_spectrum",
    ):
        values = np.asarray(result[key])
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"{key} is not a finite 1D array in {path}")
    return result


def display_spectra(
    case: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use the exact peak as unit intensity and area-match the SCBA curve."""

    exact_energy = np.asarray(case["exact_energy"], dtype=float)
    exact = np.asarray(case["exact_spectrum"], dtype=float)
    scba_energy = np.asarray(case["scba_energy"], dtype=float)
    scba = np.asarray(case["scba_spectrum"], dtype=float)
    exact_scale = float(np.max(exact))
    if exact_scale <= 0.0:
        raise RuntimeError(f"Nonpositive exact peak in {case['path']}")
    exact_display = exact / exact_scale
    scba_display = match_area_to_reference(
        exact_energy,
        exact,
        scba_energy,
        scba,
    ) / exact_scale
    return exact_energy, exact_display, scba_energy, scba_display


def style_axis(axis: plt.Axes) -> None:
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(*Y_LIMITS)
    axis.xaxis.set_major_locator(FixedLocator((1.8, 2.0, 2.2)))
    axis.xaxis.set_minor_locator(MultipleLocator(0.05))
    axis.yaxis.set_major_locator(MultipleLocator(0.5))
    axis.yaxis.set_minor_locator(MultipleLocator(0.1))
    apply_shared_axis_style(axis)


def row_label(n_molecules: int | None) -> str:
    if n_molecules is None:
        return "Outside cavity"
    return rf"$N = {n_molecules}$"


def plot_figure() -> tuple[Path, Path, Path]:
    """Create FigS4.pdf/png and a machine-readable figure audit."""

    calculation_validation = PROTOCOL_ROOT / "calculation_validation.txt"
    if (
        not calculation_validation.is_file()
        or "status = PASS"
        not in calculation_validation.read_text(encoding="utf-8")
    ):
        raise RuntimeError("20-meV calculation validation is absent or failed")

    OUTPUT_ROOT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(
        5,
        3,
        figsize=(6.95, 7.25),
        sharex=True,
        sharey=True,
    )
    grid_left = 0.088
    grid_right = 0.978
    grid_bottom = 0.073
    grid_top = 0.885
    fig.subplots_adjust(
        left=grid_left,
        right=grid_right,
        bottom=grid_bottom,
        top=grid_top,
        hspace=0.26,
        wspace=0.12,
    )
    loaded_paths: list[Path] = []
    maximum_displayed = 0.0

    for row, n_molecules in enumerate(ROW_CASES):
        for column, s_value in enumerate(HUANG_RHYS):
            case = load_case(n_molecules, s_value)
            loaded_paths.append(case["path"])  # type: ignore[arg-type]
            (
                exact_energy,
                exact_display,
                scba_energy,
                scba_display,
            ) = display_spectra(case)
            exact_mask = (
                (exact_energy >= X_LIMITS[0])
                & (exact_energy <= X_LIMITS[1])
            )
            scba_mask = (
                (scba_energy >= X_LIMITS[0])
                & (scba_energy <= X_LIMITS[1])
            )
            maximum_displayed = max(
                maximum_displayed,
                float(np.max(exact_display[exact_mask])),
                float(np.max(scba_display[scba_mask])),
            )

            axis = axes[row, column]
            axis.plot(
                exact_energy[exact_mask],
                exact_display[exact_mask],
                label="Exact",
                **EXACT_STYLE,
            )
            axis.plot(
                scba_energy[scba_mask],
                scba_display[scba_mask],
                label="SCBA",
                **SCBA_STYLE,
            )
            add_panel_label(axis, PANEL_LABELS[3 * row + column])
            style_axis(axis)

        axes[row, 0].annotate(
            row_label(n_molecules),
            xy=(1.0, 1.0),
            xycoords="axes fraction",
            xytext=(-1.0, 4.0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8.1,
            color="black",
            zorder=9,
            annotation_clip=False,
            clip_on=False,
        )

    for column, s_value in enumerate(HUANG_RHYS):
        axes[0, column].set_title(
            rf"$S = {s_value:g}$",
            pad=13.0,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=((grid_left + grid_right) / 2.0, 0.997),
        **compact_legend_kwargs(),
    )
    fig.supxlabel(
        "Energy (eV)",
        x=(grid_left + grid_right) / 2.0,
        y=0.010,
    )
    fig.supylabel("Intensity", x=0.012, y=(grid_bottom + grid_top) / 2.0)

    pdf = OUTPUT_ROOT / "FigS4.pdf"
    png = OUTPUT_ROOT / "FigS4.png"
    fig.savefig(
        pdf,
        metadata={
            "Title": "Discrete-mode cavity benchmark at 0-0 resonance",
            "Creator": "Figures/SI/FigS4/plot_FigS4.py",
        },
    )
    fig.savefig(png, dpi=220)
    plt.close(fig)

    if len(loaded_paths) != 15 or len(set(loaded_paths)) != 15:
        raise RuntimeError("The figure did not consume 15 distinct cases")
    if maximum_displayed >= Y_LIMITS[1]:
        raise RuntimeError(
            f"A displayed curve ({maximum_displayed:.6g}) exceeds the y limit"
        )

    validation = FIGURE_ROOT / "figure_validation.txt"
    validation.write_text(
        "\n".join(
            [
                "FigS4 fixed-g_c figure validation",
                "====================================",
                "status = PASS",
                "layout = 5 rows x 3 columns",
                "figure size = 6.95 x 7.25 inches",
                "axes-grid horizontal extent = 0.088 .. 0.978 (89.0% of artboard)",
                "row labels = upper-right in reserved band above first-column axes",
                "shared labels/legend centered on axes grid",
                "bottom major ticks = 1.8, 2.0, 2.2 eV (display range unchanged)",
                "column order = S=0.5, 1.0, 2.0",
                "row order = outside cavity, inside N=1, 2, 4, 8",
                "panel labels = (a) through (o), row major",
                "curves per panel = Exact and SCBA",
                "phonon frequency = 0.020 eV",
                "broadening = 0.0005 eV; temperature = 0",
                "cavity resonance = omega_c = omega_00 = 2.0 eV",
                "single-molecule coupling = g_c = 0.03536 eV",
                "collective coupling = sqrt(N) g_c (increases with N)",
                "normalization = exact peak; SCBA area matched over 1.8--2.3 eV",
                "ED cutoff = Q=28; checked at Q=30",
                "Lanczos production/check = 5000/4000 steps",
                "SCBA grid spacing = 0.05 meV",
                "curves = unsimplified vector paths",
                "style = Figures/common/si_plot_style.py",
                f"plotted cases = {len(loaded_paths)}",
                f"maximum displayed ordinate = {maximum_displayed:.12e}",
                f"shared y limit = {Y_LIMITS[0]} .. {Y_LIMITS[1]}",
                "calculation validation = PASS",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return pdf, png, validation


def main() -> None:
    configure_si_matplotlib()
    outputs = plot_figure()
    for output in outputs:
        print(f"Saved {output.relative_to(PROJECT_ROOT)}")
    print("Figure validation: PASS")


if __name__ == "__main__":
    main()
