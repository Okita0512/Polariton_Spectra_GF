#!/usr/bin/env python3
"""Generate Fig. S9: nine signed-J Drude-dimer HEOM/SCBA spectra.

The manuscript figure contains only the 3 x 3 area-normalized spectral
montage.  The command-18 W1 calculation is preserved under ``results`` as an
explicitly unplotted diagnostic and is not recomputed by this driver.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Figures.common.dimer.drude.benchmark import (  # noqa: E402
    NormalizedCase,
    load_sources,
    normalize_case,
    write_area_validation_csv,
    write_area_validation_report,
)
from Figures.common.si_plot_style import (  # noqa: E402
    HEOM_BLUE,
    SCBA_STYLE,
    add_panel_label,
    compact_legend_kwargs,
    configure_si_matplotlib,
    style_axis,
)


DISPLAY_WINDOW_EV = (1.80, 2.30)


def plot_montage(path: Path, cases: list[NormalizedCase]) -> None:
    """Draw exactly nine spectra, with no W1 axis or peak-shift annotation."""

    if len(cases) != 9:
        raise RuntimeError(f"Fig. S9 requires nine cases, received {len(cases)}")
    configure_si_matplotlib()
    figure, axes = plt.subplots(
        3,
        3,
        # Match the native PDF width to 0.80 of the 510-pt SI line width.
        # LaTeX therefore applies essentially unit scale, preserving the
        # shared 8--10 pt typography in the compact two-figure page layout.
        figsize=(5.65, 5.02),
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.095,
        top=0.900,
        wspace=0.13,
        hspace=0.23,
    )

    global_max = 0.0
    for case in cases:
        display = (case.energy_eV >= DISPLAY_WINDOW_EV[0]) & (
            case.energy_eV <= DISPLAY_WINDOW_EV[1]
        )
        global_max = max(
            global_max,
            float(np.max(case.heom_normalized_per_eV[display])),
            float(np.max(case.scba_normalized_per_eV[display])),
        )

    for index, (axis, case) in enumerate(zip(axes.ravel(), cases)):
        display = (case.energy_eV >= DISPLAY_WINDOW_EV[0]) & (
            case.energy_eV <= DISPLAY_WINDOW_EV[1]
        )
        axis.fill_between(
            case.energy_eV[display],
            0.0,
            case.heom_normalized_per_eV[display],
            color=HEOM_BLUE,
            alpha=0.28,
            linewidth=0.0,
            zorder=1,
        )
        axis.plot(
            case.energy_eV[display],
            case.heom_normalized_per_eV[display],
            color="#277DA1",
            linestyle="-",
            linewidth=0.90,
            label="HEOM",
            zorder=2,
        )
        axis.plot(
            case.energy_eV[display],
            case.scba_normalized_per_eV[display],
            label="SCBA",
            **SCBA_STYLE,
        )
        axis.set_xlim(*DISPLAY_WINDOW_EV)
        axis.set_ylim(0.0, 1.06 * global_max)
        axis.set_xticks((1.8, 2.0, 2.2))
        axis.minorticks_on()
        style_axis(axis)
        add_panel_label(axis, f"({chr(ord('a') + index)})")
        axis.annotate(
            rf"$J={case.j_mev:g}$ meV",
            xy=(1.0, 1.0),
            xycoords="axes fraction",
            xytext=(-1.0, 4.0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8.0,
            color="black",
            annotation_clip=False,
            clip_on=False,
        )

    figure.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.988),
        ncol=2,
        **compact_legend_kwargs(),
    )
    figure.text(0.54, 0.028, "Energy (eV)", ha="center", va="center", fontsize=9.5)
    figure.text(
        0.018,
        0.51,
        r"Normalized $A(E)$ (eV$^{-1}$)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=9.5,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="pdf",
        metadata={
            "Title": "Signed-J Drude-dimer HEOM/SCBA benchmark",
            "Creator": "Figures/SI/FigS7/plot_FigS7.py",
        },
    )
    plt.close(figure)


def main() -> None:
    figure_root = FIGURE_DIR
    manifest_path = figure_root / "data_manifest.csv"
    results_root = figure_root / "results"
    figure_path = figure_root / "FigS7.pdf"

    source_cases = load_sources(PROJECT_ROOT, manifest_path)
    normalized_cases = [normalize_case(case) for case in source_cases]
    write_area_validation_csv(
        results_root / "area_normalization_validation.csv",
        normalized_cases,
        DISPLAY_WINDOW_EV,
    )
    write_area_validation_report(
        results_root / "area_normalization_validation.txt",
        normalized_cases,
        DISPLAY_WINDOW_EV,
    )
    plot_montage(figure_path, normalized_cases)

    print(f"Wrote {figure_path.relative_to(PROJECT_ROOT)}")
    print("Wrote Figures/SI/FigS7/results/area_normalization_validation.csv")
    print("Wrote Figures/SI/FigS7/results/area_normalization_validation.txt")
    print("spectral_panels=9 panel_j=false W1_plotted=false")


if __name__ == "__main__":
    main()
