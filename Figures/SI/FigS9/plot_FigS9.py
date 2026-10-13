#!/usr/bin/env python3
"""Plot the compact single-molecule BA/SCBA convergence diagnostic."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np


EV_PER_HARTREE = 27.211397
SCBA_TOLERANCE_HARTREE = 1.0e-8
SPECTRUM_WINDOW_EV = (1.8, 2.25)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = SCRIPT_DIR / "N=1"
sys.path.insert(0, str(PROJECT_ROOT / "Figures" / "common"))

from area_normalization import match_area_to_reference  # noqa: E402
from si_plot_style import (  # noqa: E402
    BA_STYLE,
    HEOM_BLUE,
    SCBA_STYLE,
    add_panel_label,
    compact_legend_kwargs,
    configure_si_matplotlib,
    style_axis,
)


def load_spectra() -> tuple[np.ndarray, ...]:
    heom_energy = np.loadtxt(DATA_DIR / "resp1st.w1") * EV_PER_HARTREE
    heom_spectrum = np.loadtxt(DATA_DIR / "resp1st_im.w") / EV_PER_HARTREE

    ba = np.loadtxt(DATA_DIR / "Aw_single_shot_ba_fullpol.txt")
    ba_energy = ba[:, 0] * EV_PER_HARTREE
    ba_spectrum = match_area_to_reference(
        heom_energy,
        heom_spectrum,
        ba_energy,
        -ba[:, 1] / EV_PER_HARTREE,
    )

    scba = np.loadtxt(DATA_DIR / "Aw_sc_fullpol.txt")
    scba_energy = scba[:, 0] * EV_PER_HARTREE
    scba_spectrum = match_area_to_reference(
        heom_energy,
        heom_spectrum,
        scba_energy,
        -scba[:, 1] / EV_PER_HARTREE,
    )
    return (
        heom_energy,
        heom_spectrum,
        ba_energy,
        ba_spectrum,
        scba_energy,
        scba_spectrum,
    )


def load_self_energies() -> tuple[np.ndarray, np.ndarray]:
    ba_sigma = np.loadtxt(DATA_DIR / "Sigma_single_shot_ba_fullpol.txt")
    scba_sigma = np.loadtxt(DATA_DIR / "Sigma_sc_fullpol.txt")
    if ba_sigma.shape[1] != 3 or scba_sigma.shape[1] != 3:
        raise ValueError("Self-energy tables must contain omega, Re Sigma, Im Sigma")
    return ba_sigma, scba_sigma


def load_full_grid_residual() -> tuple[np.ndarray, np.ndarray]:
    convergence = np.loadtxt(DATA_DIR / "scba_convergence.txt")
    if convergence.ndim != 2 or convergence.shape[1] < 5:
        raise ValueError("scba_convergence.txt must contain the five recorded columns")
    iterations = convergence[:, 0]
    residual = convergence[:, 3]
    mask = (iterations > 0) & np.isfinite(residual)
    iterations = iterations[mask]
    residual = residual[mask]
    if len(iterations) == 0 or np.any(residual <= 0):
        raise ValueError("No positive full-grid residual history was recorded")
    if not np.all(np.diff(iterations) > 0):
        raise ValueError("SCBA iteration indices are not strictly increasing")
    if not residual[-1] < SCBA_TOLERANCE_HARTREE:
        raise RuntimeError("Recorded SCBA history does not reach the stated tolerance")
    if np.any(residual[:-1] < SCBA_TOLERANCE_HARTREE):
        raise RuntimeError("Recorded SCBA history continues after first convergence")
    return iterations, residual


def main() -> None:
    configure_si_matplotlib()
    (
        heom_energy,
        heom_spectrum,
        ba_energy,
        ba_spectrum,
        scba_energy,
        scba_spectrum,
    ) = load_spectra()
    ba_sigma, scba_sigma = load_self_energies()
    iterations, residual = load_full_grid_residual()

    # The explicit top margin is a real label band: panel letters and the
    # self-energy key sit above, rather than on top of, the plotted curves.
    figure = plt.figure(figsize=(7.25, 3.32))
    outer = figure.add_gridspec(
        1,
        3,
        width_ratios=(1.05, 1.15, 0.92),
        wspace=0.30,
    )
    ax_spectrum = figure.add_subplot(outer[0, 0])
    sigma_grid = outer[0, 1].subgridspec(2, 1, hspace=0.06)
    ax_real = figure.add_subplot(sigma_grid[0, 0])
    ax_imag = figure.add_subplot(sigma_grid[1, 0], sharex=ax_real)
    ax_residual = figure.add_subplot(outer[0, 2])

    ax_spectrum.fill_between(
        heom_energy,
        heom_spectrum,
        0.0,
        color=HEOM_BLUE,
        alpha=0.30,
        linewidth=0.0,
        label="HEOM",
    )
    ax_spectrum.plot(
        ba_energy,
        ba_spectrum,
        label="Single-shot BA",
        **BA_STYLE,
    )
    ax_spectrum.plot(
        scba_energy,
        scba_spectrum,
        label="SCBA",
        **SCBA_STYLE,
    )
    ax_spectrum.set_xlim(*SPECTRUM_WINDOW_EV)
    ax_spectrum.set_ylim(-2.0, 72.0)
    ax_spectrum.xaxis.set_major_locator(MultipleLocator(0.1))
    ax_spectrum.xaxis.set_minor_locator(MultipleLocator(0.02))
    ax_spectrum.yaxis.set_major_locator(MultipleLocator(20.0))
    ax_spectrum.yaxis.set_minor_locator(MultipleLocator(10.0))
    ax_spectrum.set_xlabel(r"$\omega$ (eV)")
    ax_spectrum.set_ylabel(r"$A(\omega)$ (eV$^{-1}$)")
    ax_spectrum.legend(loc="upper right", **compact_legend_kwargs())

    for table, line_style, label in (
        (ba_sigma, BA_STYLE, "Single-shot BA"),
        (scba_sigma, SCBA_STYLE, "SCBA"),
    ):
        energy = table[:, 0] * EV_PER_HARTREE
        ax_real.plot(
            energy,
            table[:, 1] * EV_PER_HARTREE * 1000.0,
            label=label,
            **line_style,
        )
        ax_imag.plot(
            energy,
            table[:, 2] * EV_PER_HARTREE * 1000.0,
            **line_style,
        )
    for axis in (ax_real, ax_imag):
        axis.set_xlim(*SPECTRUM_WINDOW_EV)
        axis.xaxis.set_major_locator(MultipleLocator(0.1))
        axis.xaxis.set_minor_locator(MultipleLocator(0.02))
        axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax_real.set_ylim(-35.0, 20.0)
    ax_imag.set_ylim(-45.0, 10.0)
    ax_real.tick_params(labelbottom=False)
    ax_real.set_ylabel(r"$\mathrm{Re}\,\Sigma^R$ (meV)")
    ax_imag.set_ylabel(r"$\mathrm{Im}\,\Sigma^R$ (meV)")
    ax_imag.set_xlabel(r"$\omega$ (eV)")
    ax_real.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.12),
        ncol=2,
        **compact_legend_kwargs(),
    )

    ax_residual.semilogy(
        iterations,
        residual,
        "o-",
        color="0.20",
        linewidth=0.90,
        markersize=3.0,
        markerfacecolor="white",
        markeredgecolor="0.20",
        markeredgewidth=0.65,
    )
    ax_residual.axhline(
        SCBA_TOLERANCE_HARTREE,
        color="0.45",
        linestyle=(0, (3.0, 2.0)),
        linewidth=0.80,
    )
    ax_residual.set_xlim(0.50, iterations[-1] + 0.35)
    ax_residual.set_ylim(3.0e-10, 3.0e-3)
    ax_residual.xaxis.set_major_locator(MultipleLocator(1.0))
    ax_residual.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax_residual.set_xlabel("SCBA iteration")
    ax_residual.set_ylabel(r"$\epsilon_\Sigma$ ($E_h$)")
    ax_residual.text(
        0.72,
        1.24 * SCBA_TOLERANCE_HARTREE,
        "Tolerance",
        fontsize=7.5,
        color="0.30",
        ha="left",
        va="bottom",
    )

    for axis in (ax_spectrum, ax_real, ax_imag, ax_residual):
        style_axis(axis)
    add_panel_label(ax_spectrum, "(a)")
    add_panel_label(ax_real, "(b)")
    add_panel_label(ax_residual, "(c)")

    figure.subplots_adjust(
        left=0.078,
        right=0.990,
        bottom=0.175,
        top=0.825,
        wspace=0.37,
    )
    output = SCRIPT_DIR / "FigS9.pdf"
    figure.savefig(output, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)
    print(
        f"Saved {output.relative_to(PROJECT_ROOT)}; final full-grid residual "
        f"= {residual[-1]:.6e} E_h at iteration {int(iterations[-1])}"
    )


if __name__ == "__main__":
    main()
