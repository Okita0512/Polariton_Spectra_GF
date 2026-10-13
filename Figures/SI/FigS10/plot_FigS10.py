#!/usr/bin/env python3
"""Plot the SI detuning spectra and upper-polariton linewidth benchmark.

The numerical loaders remain in the shared detuning module used by the frozen
main-text figure. This local renderer applies the SI-only style without
changing that shared renderer or any cached data.
"""

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
SHARED_DIR = PROJECT_ROOT / "Figures" / "shared" / "detuning"
COMMON_DIR = PROJECT_ROOT / "Figures" / "common"
sys.path.insert(0, str(SHARED_DIR))
sys.path.insert(0, str(COMMON_DIR))

from detuning_plot import (  # noqa: E402
    CAVITY_FWHM_MEV,
    N_VALUES,
    PHYSICAL_CAVITY_EV,
    SPECTRUM_CACHE_KEYS,
    input_checksum,
    load_linewidth_series,
    load_offset_spectrum,
    write_validation,
)
from endpoint_linewidths import load_endpoint_linewidths  # noqa: E402
from si_plot_style import (  # noqa: E402
    HEOM_BLUE,
    N_COLORS,
    SCBA_STYLE,
    add_panel_label,
    compact_legend_kwargs,
    configure_si_matplotlib,
    style_axis,
)


def n_label(n_value: int) -> str:
    return r"$N\to\infty$" if n_value == 10000 else rf"$N={n_value}$"


def benchmark_method(n_value: int) -> str | None:
    if n_value in (1, 2, 4, 8):
        return "HEOM"
    if n_value == 10000:
        return "MF-HEOM"
    return None


def plot_up_linewidths(axis: plt.Axes) -> tuple[list[Line2D], list[Line2D]]:
    """Plot every retained N series with method encoded by marker fill."""

    endpoints = load_endpoint_linewidths()
    n_handles: list[Line2D] = []
    for n_value in N_VALUES:
        color = N_COLORS[n_value]
        scba = load_linewidth_series(n_value, "SCBA", "up")
        axis.plot(
            np.concatenate(([0.0], scba.exciton_fraction, [1.0])),
            np.concatenate(
                (
                    [CAVITY_FWHM_MEV],
                    scba.linewidth_mev,
                    [endpoints.scba_molecule_fwhm_meV],
                )
            ),
            color=color,
            linewidth=1.00,
            marker="o",
            markersize=2.2,
            markeredgewidth=0.0,
            zorder=2,
        )
        n_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=1.35,
                marker="o",
                markersize=2.4,
                markeredgewidth=0.0,
                label=n_label(n_value),
            )
        )

        method = benchmark_method(n_value)
        if method is None:
            continue
        benchmark = load_linewidth_series(n_value, method, "up")
        axis.plot(
            np.concatenate(([0.0], benchmark.exciton_fraction, [1.0])),
            np.concatenate(
                (
                    [CAVITY_FWHM_MEV],
                    benchmark.linewidth_mev,
                    [endpoints.heom_molecule_fwhm_meV],
                )
            ),
            linestyle="none",
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=0.85,
            zorder=4,
        )

    fraction = np.linspace(0.0, 1.0, 401)
    hopfield = (
        (1.0 - fraction) * CAVITY_FWHM_MEV
        + fraction * endpoints.eq26_gamma_ex_meV
    )
    axis.plot(
        fraction,
        hopfield,
        color="0.62",
        linewidth=1.8,
        zorder=1,
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(30.0, 110.0)
    axis.xaxis.set_major_locator(MultipleLocator(0.2))
    axis.xaxis.set_minor_locator(MultipleLocator(0.1))
    axis.yaxis.set_major_locator(MultipleLocator(20.0))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.set_xlabel(r"Exciton fraction $|C_{\mathrm{ex},+}|^2$")
    axis.set_ylabel(r"$\Gamma_{\mathrm{UP}}$ (meV)")

    method_handles = [
        Line2D(
            [0], [0], color="black", linewidth=1.0, marker="o",
            markersize=2.5, markeredgewidth=0.0, label="SCBA",
        ),
        Line2D(
            [0], [0], color="black", linestyle="none", marker="o",
            markersize=3.7, markerfacecolor="white", markeredgewidth=0.85,
            label="HEOM / MF-HEOM",
        ),
        Line2D(
            [0], [0], color="0.62", linewidth=1.8,
            label="Hopfield-weighted",
        ),
    ]
    return n_handles, method_handles


def main() -> None:
    configure_si_matplotlib()
    figure = plt.figure(figsize=(7.25, 3.35))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.08), wspace=0.29)
    spectra_axis = figure.add_subplot(grid[0, 0])
    up_axis = figure.add_subplot(grid[0, 1])

    spectrum_pairs = [
        load_offset_spectrum(cache_key) for cache_key in SPECTRUM_CACHE_KEYS
    ]
    largest_display_peak = max(
        max(float(np.max(pair[1])), float(np.max(pair[3])))
        for pair in spectrum_pairs
    )
    offset_step = largest_display_peak + 0.08
    area_factors: dict[str, float] = {}
    area_errors: list[float] = []
    for index, (cavity_frequency, pair) in enumerate(
        zip(PHYSICAL_CAVITY_EV, spectrum_pairs)
    ):
        heom_energy, heom_display, scba_energy, scba_display, factor = pair
        offset = index * offset_step
        spectra_axis.fill_between(
            heom_energy,
            offset,
            heom_display + offset,
            color=HEOM_BLUE,
            alpha=0.32,
            linewidth=0.0,
            zorder=1,
        )
        spectra_axis.plot(
            scba_energy,
            scba_display + offset,
            **SCBA_STYLE,
        )
        spectra_axis.text(
            1.515,
            offset + 0.68,
            rf"$\omega_\mathrm{{c}}={cavity_frequency:.2f}$ eV",
            fontsize=7.2,
            ha="left",
            va="center",
        )

        area_factors[f"omega_c={cavity_frequency:.2f} eV"] = factor
        overlap_min = max(float(np.min(heom_energy)), float(np.min(scba_energy)))
        overlap_max = min(float(np.max(heom_energy)), float(np.max(scba_energy)))
        heom_mask = (heom_energy >= overlap_min) & (heom_energy <= overlap_max)
        scba_mask = (scba_energy >= overlap_min) & (scba_energy <= overlap_max)
        heom_area = float(np.trapz(heom_display[heom_mask], heom_energy[heom_mask]))
        scba_area = float(np.trapz(scba_display[scba_mask], scba_energy[scba_mask]))
        area_errors.append(abs(scba_area / heom_area - 1.0))

    spectra_axis.set_xlim(1.5, 2.5)
    spectra_axis.set_ylim(
        -0.18,
        (len(SPECTRUM_CACHE_KEYS) - 1) * offset_step
        + largest_display_peak
        + 0.04,
    )
    spectra_axis.set_xlabel(r"$\omega$ (eV)")
    spectra_axis.set_ylabel("Normalized absorption + offset")
    spectra_axis.set_yticks([])
    spectra_axis.xaxis.set_major_locator(MultipleLocator(0.2))
    spectra_axis.xaxis.set_minor_locator(MultipleLocator(0.1))
    spectra_axis.legend(
        handles=(
            Patch(
                facecolor=HEOM_BLUE,
                alpha=0.32,
                edgecolor="none",
                label="HEOM",
            ),
            Line2D([0], [0], label="SCBA", **SCBA_STYLE),
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        **compact_legend_kwargs(),
    )

    n_handles, method_handles = plot_up_linewidths(up_axis)
    n_legend = up_axis.legend(
        handles=n_handles,
        title="Ensemble size",
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        title_fontsize=8.1,
        **compact_legend_kwargs(),
    )
    up_axis.add_artist(n_legend)
    up_axis.legend(
        handles=method_handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.67),
        **compact_legend_kwargs(),
    )

    for axis in (spectra_axis, up_axis):
        style_axis(axis)
    add_panel_label(spectra_axis, "(a)")
    add_panel_label(up_axis, "(b)")

    figure.subplots_adjust(left=0.078, right=0.990, bottom=0.165, top=0.905)
    output = SCRIPT_DIR / "FigS10.pdf"
    figure.savefig(output, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)

    count, checksum = input_checksum()
    validation = {
        "role": "SI detuning spectra and UP linewidth",
        "output": output.relative_to(PROJECT_ROOT).as_posix(),
        "active_input_count": count,
        "active_input_sha256": checksum,
        "offset_spectrum_pairs": len(SPECTRUM_CACHE_KEYS),
        "largest_normalized_pair_peak": largest_display_peak,
        "pair_offset_step": offset_step,
        "minimum_interpair_clearance": offset_step - largest_display_peak,
        "maximum_area_ratio_error": max(area_errors),
        "scba_up_curves": len(N_VALUES),
        "benchmark_up_series": 5,
        "scba_area_match_factors": area_factors,
        "method_style": "HEOM light-blue fill; SCBA red dashed",
        "validation": "PASS",
    }
    write_validation(SCRIPT_DIR / "validation.json", validation)
    print(f"Wrote {output.relative_to(PROJECT_ROOT)}")
    print(f"Validation: {validation['validation']}")


if __name__ == "__main__":
    main()

