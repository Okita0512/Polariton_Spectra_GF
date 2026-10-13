#!/usr/bin/env python3
"""Render the collective-limit MF-HEOM/SCBA benchmark from cached data."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
COMMON_DIR = PROJECT_ROOT / "Figures" / "common"
sys.path.insert(0, str(COMMON_DIR))

from endpoint_linewidths import load_endpoint_linewidths  # noqa: E402
from si_plot_style import (  # noqa: E402
    HEOM_BLUE,
    SCBA_STYLE,
    add_panel_label,
    compact_legend_kwargs,
    configure_si_matplotlib,
    style_axis,
)


EV_PER_HARTREE = 27.2114
PROFILE_WINDOW_EV = (1.6, 2.4)
CAVITY_FREQUENCIES_EV = (1.80, 1.85, 1.90, 1.95, 2.00, 2.05, 2.10, 2.15, 2.20)
LP_COLOR = "#2468D8"
UP_COLOR = "#D62728"
REFERENCE_COLOR = "0.60"


def area_normalize(x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the retained positive, displayed-window area normalization."""

    positive = np.maximum(y_data, 0.0)
    mask = (x_data >= PROFILE_WINDOW_EV[0]) & (x_data <= PROFILE_WINDOW_EV[1])
    area = np.trapz(positive[mask], x_data[mask])
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("cannot area-normalize a nonpositive spectrum")
    return positive / area, mask


def load_mf_susceptibility(path: Path) -> np.ndarray:
    """Load the processed -Im[P/E] observable and reject stale power spectra."""

    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
    if "MinusImChi_eV^-1" not in header:
        raise ValueError(
            "stale MF-HEOM observable; run Figures/common/process_mf_heom.py"
        )
    return np.loadtxt(path)


def plot_stacked_spectra(axis: plt.Axes) -> None:
    """Plot the nine retained paired profiles without parameter rainbow colors."""

    offset_step = 1.10
    for index, cavity_frequency in enumerate(CAVITY_FREQUENCIES_EV):
        directory = SCRIPT_DIR / f"wc={cavity_frequency}"
        mf_table = load_mf_susceptibility(directory / "absorption_pulse_2eV.txt")
        # Retain the original time-window slice used by this publication plot.
        mf_energy = mf_table[:2903, 0]
        mf_spectrum = mf_table[:2903, 1]

        scba_table = np.loadtxt(directory / "Aw_sc_fullpol.txt")
        scba_energy = scba_table[:, 0] * EV_PER_HARTREE
        scba_spectrum = -scba_table[:, 1] / EV_PER_HARTREE

        mf_spectrum, mf_mask = area_normalize(mf_energy, mf_spectrum)
        scba_spectrum, scba_mask = area_normalize(scba_energy, scba_spectrum)
        pair_scale = max(
            float(np.max(mf_spectrum[mf_mask])),
            float(np.max(scba_spectrum[scba_mask])),
        )
        mf_spectrum /= pair_scale
        scba_spectrum /= pair_scale

        offset = index * offset_step
        axis.fill_between(
            mf_energy,
            offset,
            mf_spectrum + offset,
            color=HEOM_BLUE,
            alpha=0.34,
            linewidth=0.0,
            zorder=1,
        )
        axis.plot(
            scba_energy,
            scba_spectrum + offset,
            **SCBA_STYLE,
        )
        axis.text(
            1.61,
            offset + 0.67,
            rf"$\omega_\mathrm{{c}}={cavity_frequency:.2f}$ eV",
            fontsize=7.3,
            ha="left",
            va="center",
        )

    axis.set_xlim(*PROFILE_WINDOW_EV)
    axis.set_ylim(-0.45, (len(CAVITY_FREQUENCIES_EV) - 1) * offset_step + 1.08)
    axis.xaxis.set_major_locator(MultipleLocator(0.2))
    axis.xaxis.set_minor_locator(MultipleLocator(0.1))
    axis.set_yticks([])
    axis.set_xlabel(r"$\omega$ (eV)")
    axis.set_ylabel("Normalized absorption + offset")
    axis.legend(
        handles=(
            Patch(
                facecolor=HEOM_BLUE,
                alpha=0.34,
                edgecolor="none",
                label="MF-HEOM",
            ),
            Line2D([0], [0], label="SCBA", **SCBA_STYLE),
        ),
        loc="lower right",
        bbox_to_anchor=(1.0, 1.09),
        ncol=2,
        **compact_legend_kwargs(),
    )


def hopfield_references() -> tuple[np.ndarray, ...]:
    """Return the retained Hopfield-weighted reference curves in meV."""

    endpoints = load_endpoint_linewidths()
    cavity_fwhm_ev = 20.0e-3
    exciton_fwhm_ev = endpoints.eq26_gamma_ex_meV * 1.0e-3
    molecular_energy_ev = 2.03
    collective_coupling_ev = 0.10

    cavity_frequency = np.linspace(1.0, 3.0, 1000)
    detuning = cavity_frequency - molecular_energy_ev
    denominator = np.sqrt(detuning**2 + 4.0 * collective_coupling_ev**2)
    photon_up = 0.5 * (1.0 + detuning / denominator)
    exciton_up = 0.5 * (1.0 - detuning / denominator)

    linewidth_up = (
        photon_up * cavity_fwhm_ev + exciton_up * exciton_fwhm_ev
    ) * 1000.0
    linewidth_lp = (
        exciton_up * cavity_fwhm_ev + photon_up * exciton_fwhm_ev
    ) * 1000.0
    return (
        cavity_frequency,
        photon_up,
        exciton_up,
        linewidth_lp,
        linewidth_up,
    )


def load_linewidth_tables() -> tuple[np.ndarray, np.ndarray]:
    mf_table = np.loadtxt(SCRIPT_DIR / "width-MF.txt", dtype=float)
    scba_table = np.loadtxt(SCRIPT_DIR / "width-SCBA.txt", dtype=float)
    for name, table in (("MF-HEOM", mf_table), ("SCBA", scba_table)):
        if table.ndim != 2 or table.shape[1] != 3 or not np.all(np.isfinite(table)):
            raise ValueError(f"Malformed {name} linewidth table")
    if not np.array_equal(mf_table[:, 0], scba_table[:, 0]):
        raise ValueError("MF-HEOM and SCBA cavity-frequency grids differ")
    return mf_table, scba_table


def plot_linewidths(axis_wc: plt.Axes, axis_fraction: plt.Axes) -> None:
    """Plot exact eV-to-meV conversions of the cached linewidth columns."""

    mf_table, scba_table = load_linewidth_tables()
    cavity_frequency, photon_up, exciton_up, ref_lp, ref_up = hopfield_references()

    # Panel (b): linewidth versus cavity frequency.
    axis_wc.plot(
        mf_table[:, 0],
        1000.0 * mf_table[:, 1],
        linestyle="none",
        marker="o",
        markersize=3.6,
        markerfacecolor="white",
        markeredgecolor=LP_COLOR,
        markeredgewidth=0.85,
        zorder=4,
    )
    axis_wc.plot(
        mf_table[:, 0],
        1000.0 * mf_table[:, 2],
        linestyle="none",
        marker="o",
        markersize=3.6,
        markerfacecolor="white",
        markeredgecolor=UP_COLOR,
        markeredgewidth=0.85,
        zorder=4,
    )
    axis_wc.plot(
        scba_table[:, 0], 1000.0 * scba_table[:, 1],
        color=LP_COLOR, linewidth=1.0, zorder=3,
    )
    axis_wc.plot(
        scba_table[:, 0], 1000.0 * scba_table[:, 2],
        color=UP_COLOR, linewidth=1.0, zorder=3,
    )
    axis_wc.plot(
        cavity_frequency, ref_lp,
        color=LP_COLOR, linestyle=(0, (4.0, 2.0)), linewidth=1.1,
        alpha=0.45, zorder=1,
    )
    axis_wc.plot(
        cavity_frequency, ref_up,
        color=UP_COLOR, linestyle=(0, (4.0, 2.0)), linewidth=1.1,
        alpha=0.45, zorder=1,
    )

    # Panel (c): the same values against the appropriate exciton fraction.
    detuning_data = mf_table[:, 0] - 2.03
    denominator_data = np.sqrt(detuning_data**2 + 4.0 * 0.10**2)
    photon_up_data = 0.5 * (1.0 + detuning_data / denominator_data)
    exciton_up_data = 0.5 * (1.0 - detuning_data / denominator_data)
    axis_fraction.plot(
        photon_up_data,
        1000.0 * mf_table[:, 1],
        linestyle="none",
        marker="o",
        markersize=3.6,
        markerfacecolor="white",
        markeredgecolor=LP_COLOR,
        markeredgewidth=0.85,
        zorder=4,
    )
    axis_fraction.plot(
        exciton_up_data,
        1000.0 * mf_table[:, 2],
        linestyle="none",
        marker="o",
        markersize=3.6,
        markerfacecolor="white",
        markeredgecolor=UP_COLOR,
        markeredgewidth=0.85,
        zorder=4,
    )
    axis_fraction.plot(
        photon_up_data,
        1000.0 * scba_table[:, 1],
        color=LP_COLOR,
        linewidth=1.0,
        zorder=3,
    )
    axis_fraction.plot(
        exciton_up_data,
        1000.0 * scba_table[:, 2],
        color=UP_COLOR,
        linewidth=1.0,
        zorder=3,
    )
    axis_fraction.plot(
        photon_up,
        ref_lp,
        color=LP_COLOR,
        linestyle=(0, (4.0, 2.0)),
        linewidth=1.1,
        alpha=0.45,
        zorder=1,
    )
    axis_fraction.plot(
        exciton_up,
        ref_up,
        color=UP_COLOR,
        linestyle=(0, (4.0, 2.0)),
        linewidth=1.1,
        alpha=0.45,
        zorder=1,
    )

    for axis in (axis_wc, axis_fraction):
        axis.set_ylim(0.0, 110.0)
        axis.yaxis.set_major_locator(MultipleLocator(20.0))
        axis.yaxis.set_minor_locator(MultipleLocator(10.0))
        axis.set_ylabel("Linewidth (meV)")
    axis_wc.set_xlim(1.0, 3.0)
    axis_wc.xaxis.set_major_locator(MultipleLocator(0.5))
    axis_wc.xaxis.set_minor_locator(MultipleLocator(0.1))
    axis_wc.set_xlabel(r"$\omega_\mathrm{c}$ (eV)")
    axis_fraction.set_xlim(0.0, 1.0)
    axis_fraction.xaxis.set_major_locator(MultipleLocator(0.2))
    axis_fraction.xaxis.set_minor_locator(MultipleLocator(0.1))
    axis_fraction.set_xlabel(r"Exciton fraction $|C_\mathrm{ex}|^2$")

    branch_handles = (
        Line2D([0], [0], color=LP_COLOR, linewidth=1.2, label="LP"),
        Line2D([0], [0], color=UP_COLOR, linewidth=1.2, label="UP"),
    )
    axis_wc.legend(
        handles=branch_handles,
        loc="upper right",
        ncol=2,
        **compact_legend_kwargs(),
    )
    method_handles = (
        Line2D([0], [0], color="black", linewidth=1.0, label="SCBA"),
        Line2D(
            [0], [0], linestyle="none", marker="o", markersize=3.7,
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=0.85, label="MF-HEOM",
        ),
        Line2D(
            [0], [0], color=REFERENCE_COLOR,
            linestyle=(0, (4.0, 2.0)), linewidth=1.1,
            label="Hopfield-weighted",
        ),
    )
    axis_fraction.legend(
        handles=method_handles,
        loc="upper left",
        **compact_legend_kwargs(),
    )


def main() -> None:
    configure_si_matplotlib()
    figure = plt.figure(figsize=(7.25, 4.55))
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=(1.07, 1.0),
        hspace=0.48,
        wspace=0.34,
    )
    ax_a = figure.add_subplot(grid[:, 0])
    ax_b = figure.add_subplot(grid[0, 1])
    ax_c = figure.add_subplot(grid[1, 1])

    plot_stacked_spectra(ax_a)
    plot_linewidths(ax_b, ax_c)
    for axis in (ax_a, ax_b, ax_c):
        style_axis(axis)
    add_panel_label(ax_a, "(a)")
    add_panel_label(ax_b, "(b)")
    add_panel_label(ax_c, "(c)")

    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.11,
        top=0.94,
        hspace=0.48,
        wspace=0.34,
    )
    output = SCRIPT_DIR / "FigS12.pdf"
    figure.savefig(output, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    print(f"Saved {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
