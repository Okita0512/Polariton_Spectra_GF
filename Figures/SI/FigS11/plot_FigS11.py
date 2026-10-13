#!/usr/bin/env python3
"""Plot the independent bright--dark near-degeneracy diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = SCRIPT_DIR / "data"
FIGURES_DIR = SCRIPT_DIR
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT / "Figures" / "common"))

from endpoint_linewidths import load_endpoint_linewidths  # noqa: E402
from si_plot_style import (  # noqa: E402
    N_COLORS,
    TEMPERATURE_COLORS,
    add_panel_label,
    compact_legend_kwargs,
    configure_si_matplotlib,
    style_axis,
)


def read_rows(name: str) -> list[dict[str, str]]:
    with (DATA_DIR / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(
    rows: list[dict[str, str]], key: str
) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows])


def rows_for(
    rows: list[dict[str, str]],
    **conditions: object,
) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        keep = True
        for key, value in conditions.items():
            if isinstance(value, (int, float)):
                keep &= np.isclose(float(row[key]), float(value))
            else:
                keep &= row[key] == str(value)
        if keep:
            selected.append(row)
    return sorted(selected, key=lambda item: float(item["gap_meV"]))


def validate_publication_rows(
    gap_rows: list[dict[str, str]],
    temperature_rows: list[dict[str, str]],
    plotted_n: tuple[int, ...],
) -> None:
    """Fail rather than silently omit or interpolate a publication datum."""

    for n_molecules in plotted_n:
        selected = rows_for(gap_rows, N=n_molecules)
        arrays = (
            numeric(selected, "gap_meV"),
            numeric(selected, "fwhm_meV"),
        )
        if len(selected) != 12 or not all(np.all(np.isfinite(x)) for x in arrays):
            raise ValueError(f"Expected 12 finite gap/FWHM rows for N={n_molecules}")
    for n_molecules in (4, 64):
        selected = rows_for(gap_rows, N=n_molecules)
        channels = (
            numeric(selected, "minus2ImSigmaB_meV"),
            numeric(selected, "minus2ImSigmaD_meV"),
        )
        if not all(np.all(np.isfinite(x)) for x in channels):
            raise ValueError(f"Non-finite channel diagnostic for N={n_molecules}")
    for temperature in (0, 100, 200, 300):
        selected = rows_for(
            temperature_rows,
            N=64,
            temperature_K=temperature,
        )
        arrays = (
            numeric(selected, "gap_meV"),
            numeric(selected, "minus2ImSigmaD_meV"),
        )
        if len(selected) != 4 or not all(np.all(np.isfinite(x)) for x in arrays):
            raise ValueError(
                f"Expected four finite temperature rows at T={temperature} K"
            )


def main() -> None:
    configure_si_matplotlib()
    gap_rows = read_rows("gap_scan.csv")
    molecule = read_rows("isolated_molecule.csv")[0]
    temperature_rows = read_rows("temperature_scan.csv")
    endpoints = load_endpoint_linewidths()
    archived_molecule_fwhm = float(molecule["fwhm_meV"])
    molecule_fwhm = endpoints.scba_molecule_fwhm_meV
    if not np.isclose(
        archived_molecule_fwhm,
        molecule_fwhm,
        rtol=0.0,
        atol=5.0e-4,
    ):
        raise RuntimeError(
            "Fig. S13 molecular spectrum disagrees with the shared endpoint "
            "audit; rerun Figures/common/audit_endpoint_linewidths.py"
        )

    plotted_n = (1, 2, 4, 8, 16, 64, 10000)
    validate_publication_rows(gap_rows, temperature_rows, plotted_n)
    labels = {
        1: r"$N=1$",
        2: r"$N=2$",
        4: r"$N=4$",
        8: r"$N=8$",
        16: r"$N=16$",
        64: r"$N=64$",
        10000: r"$N\to\infty$",
    }

    # Extra inter-row and top space is reserved for the common panel-label
    # bands and compact keys.  No label or legend is laid over a high curve.
    figure = plt.figure(figsize=(7.25, 6.10))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.08, 1.0),
        hspace=0.58,
        wspace=0.28,
    )
    ax_a = figure.add_subplot(grid[0, :])
    ax_b = figure.add_subplot(grid[1, 0])
    ax_c = figure.add_subplot(grid[1, 1])

    pale_edges = {
        1: "#8A7900",
        2: "#718000",
    }

    for n_molecules in plotted_n:
        color = N_COLORS[n_molecules]
        selected = rows_for(gap_rows, N=n_molecules)
        gaps = numeric(selected, "gap_meV")
        widths = numeric(selected, "fwhm_meV")
        ax_a.plot(
            gaps,
            widths,
            marker="D" if n_molecules == 10000 else "o",
            linestyle="-",
            color=color,
            linewidth=1.05 if n_molecules == 10000 else 0.95,
            markersize=2.9 if n_molecules == 10000 else 2.8,
            markeredgecolor=pale_edges.get(n_molecules, color),
            markeredgewidth=0.55 if n_molecules in pale_edges else 0.0,
            label=labels[n_molecules],
        )
    ax_a.axhline(
        molecule_fwhm,
        color="0.45",
        linestyle="--",
        linewidth=0.90,
        label="Isolated molecule",
    )
    ax_a.set_xscale("log")
    ax_a.set_ylabel(r"$\Gamma_{\mathrm{UP}}^{\mathrm{SCBA}}$ (meV)")
    ax_a.set_xlabel(r"$\delta_{+D}$ (meV)")
    ax_a.set_xlim(0.015, 120.0)
    ax_a.legend(
        ncol=2,
        fontsize=7.5,
        loc="lower left",
        bbox_to_anchor=(0.012, 0.025),
        **compact_legend_kwargs(),
    )

    for n_molecules in (4, 64):
        selected = rows_for(gap_rows, N=n_molecules)
        gaps = numeric(selected, "gap_meV")
        bright = numeric(selected, "minus2ImSigmaB_meV")
        dark = numeric(selected, "minus2ImSigmaD_meV")
        color = N_COLORS[n_molecules]
        ax_b.plot(
            gaps,
            dark,
            "o-",
            color=color,
            linewidth=1.00,
            markersize=2.8,
            markeredgecolor="#4F7D18" if n_molecules == 4 else color,
            markeredgewidth=0.45 if n_molecules == 4 else 0.0,
            label=rf"$N={n_molecules}$, dark",
        )
        ax_b.plot(
            gaps,
            bright,
            "s--",
            color=color,
            linewidth=0.90,
            markersize=2.6,
            markeredgecolor="#4F7D18" if n_molecules == 4 else color,
            markeredgewidth=0.45 if n_molecules == 4 else 0.0,
            label=rf"$N={n_molecules}$, bright",
        )
    ax_b.set_xscale("log")
    ax_b.set_xlim(0.015, 120.0)
    ax_b.set_xlabel(r"$\delta_{+D}$ (meV)")
    ax_b.set_ylabel(r"$-2\,\mathrm{Im}\,\Sigma^R_{B,D}$ (meV)")
    ax_b.legend(
        fontsize=7.5,
        ncol=2,
        loc="lower left",
        bbox_to_anchor=(0.12, 1.13),
        **compact_legend_kwargs(),
    )

    temperature_styles = {
        0: "-",
        100: (0, (5.0, 1.8)),
        200: (0, (2.8, 1.5)),
        300: (0, (1.2, 1.2)),
    }
    for temperature in (0, 100, 200, 300):
        selected = rows_for(
            temperature_rows, N=64, temperature_K=temperature
        )
        gaps = numeric(selected, "gap_meV")
        dark = numeric(selected, "minus2ImSigmaD_meV")
        ax_c.plot(
            gaps,
            dark,
            marker="o",
            linestyle=temperature_styles[temperature],
            color=TEMPERATURE_COLORS[temperature],
            linewidth=0.95,
            markersize=2.9,
            markeredgewidth=0.0,
            label=rf"{temperature} K",
        )
    ax_c.set_xscale("log")
    ax_c.set_xlim(0.15, 25.0)
    ax_c.set_xlabel(r"$\delta_{+D}$ (meV)")
    ax_c.set_ylabel(r"$-2\,\mathrm{Im}\,\Sigma^R_D$ (meV)")
    ax_c.set_ylim(0.0, 70.0)
    ax_c.legend(
        fontsize=7.7,
        ncol=4,
        loc="lower left",
        bbox_to_anchor=(0.12, 1.13),
        **compact_legend_kwargs(),
    )

    for axis in (ax_a, ax_b, ax_c):
        style_axis(axis)
        axis.xaxis.set_minor_locator(
            LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
        )
    add_panel_label(ax_a, "(a)")
    add_panel_label(ax_b, "(b)")
    add_panel_label(ax_c, "(c)")

    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.080, top=0.825)
    output = FIGURES_DIR / "FigS11.pdf"
    figure.savefig(output, bbox_inches="tight")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
