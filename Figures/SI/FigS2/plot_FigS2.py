#!/usr/bin/env python3
"""Plot HEOM, exact cumulant, single-shot BA, and SCBA spectra."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator
import numpy as np


FIGURE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_ROOT.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "Figures" / "common"))
from si_plot_style import (  # noqa: E402
    BA_STYLE,
    EXACT_STYLE,
    HEOM_BLUE,
    SCBA_STYLE,
    add_panel_label,
    compact_legend_kwargs,
    configure_si_matplotlib,
    style_axis,
)


EV_PER_AU = 27.211397
LAMBDAS_MEV = (15.0, 30.0, 45.0, 60.0)
CUTOFFS_MEV = (12.4, 24.8, 37.2)


def normalize(values: np.ndarray) -> np.ndarray:
    maximum = float(np.max(values))
    return values / maximum if maximum > 0.0 else values


def match_area(
    energy: np.ndarray,
    values: np.ndarray,
    target_area: float,
) -> np.ndarray:
    """Scale a spectrum to a target area without peak normalization."""
    area = float(np.trapz(values, energy))
    if area <= 0.0:
        raise RuntimeError("Cannot area-match a nonpositive spectrum")
    return values * (target_area / area)


def electronic_path(
    results: Path,
    mode: str,
    lam: float,
    cutoff: float,
) -> Path:
    return (
        results
        / f"frenkel_{mode}_lambda_{lam:.0f}_cutoff_{cutoff:.1f}.txt"
    )


def bare_bath_spectrum(
    lam_mev: float,
    cutoff_mev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Collective Drude spectrum B_X,0(nu)=2J(nu) for G-only SCBA."""
    nu = np.linspace(-0.25, 0.25, 4001)
    lam = lam_mev * 1.0e-3
    cutoff = cutoff_mev * 1.0e-3
    spectral = (
        4.0 * lam * cutoff * nu / (nu**2 + cutoff**2)
    )
    return nu, spectral


def plot_electronic(root: Path) -> tuple[Path, Path]:
    reference = root / "heom_reference"
    results = root / "results"
    cumulant_results = root / "exact_cumulant"
    figures = root
    fig, axes = plt.subplots(
        3, 4, figsize=(7.08, 5.00), sharex=True, sharey=True
    )
    fig.subplots_adjust(
        left=0.082,
        right=0.992,
        bottom=0.105,
        top=0.875,
        wspace=0.16,
        hspace=0.23,
    )
    letters = "abcdefghijkl"

    for row, cutoff in enumerate(CUTOFFS_MEV):
        for col, lam in enumerate(LAMBDAS_MEV):
            ax = axes[row, col]
            case = reference / f"{lam:.0f}_{cutoff:.1f}"

            omega_heom = np.loadtxt(case / "resp1st.w1") * EV_PER_AU
            heom_raw = np.loadtxt(case / "resp1st_im.w")

            cumulant = np.loadtxt(
                cumulant_results
                / f"cumulant_lambda_{lam:.0f}_cutoff_{cutoff:.1f}.txt"
            )
            cumulant_plot = normalize(cumulant[:, 1])
            target_area = float(
                np.trapz(cumulant_plot, cumulant[:, 0])
            )
            heom_plot = match_area(
                omega_heom,
                heom_raw,
                target_area,
            )
            ax.fill_between(
                omega_heom,
                heom_plot,
                0.0,
                color=HEOM_BLUE,
                alpha=0.30,
                linewidth=0.0,
                label="HEOM",
            )
            ax.plot(
                cumulant[:, 0],
                cumulant_plot,
                label="Exact cumulant",
                **EXACT_STYLE,
            )

            g_only = np.loadtxt(
                electronic_path(results, "g_only", lam, cutoff)
            )
            ba = np.loadtxt(
                electronic_path(results, "single_shot_ba", lam, cutoff)
            )
            ba_plot = match_area(
                ba[:, 0],
                ba[:, 1],
                target_area,
            )
            scba_plot = match_area(
                g_only[:, 0],
                g_only[:, 1],
                target_area,
            )
            ax.plot(
                ba[:, 0],
                ba_plot,
                label="Single-shot BA",
                **BA_STYLE,
            )
            ax.plot(
                g_only[:, 0],
                scba_plot,
                label="SCBA",
                **SCBA_STYLE,
            )

            ax.set_xlim(1.8, 2.4)
            ax.set_ylim(-0.05, 1.12)
            ax.minorticks_on()
            ax.xaxis.set_major_locator(FixedLocator((1.9, 2.1, 2.3)))
            ax.xaxis.set_minor_locator(MultipleLocator(0.05))
            style_axis(ax)
            add_panel_label(ax, f"({letters[row * 4 + col]})")
            if row == 0:
                ax.set_title(
                    rf"$\lambda = {lam:.0f}$ meV", pad=3.0
                )
            if col == 0:
                ax.set_ylabel(
                    rf"$\omega_f = {cutoff:.1f}$ meV"
                    + "\nIntensity"
                )
            if row == 2:
                ax.set_xlabel("Energy (eV)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.537, 0.995),
        **compact_legend_kwargs(),
    )
    pdf = figures / "FigS2.pdf"
    png = pdf.with_suffix(".png")
    fig.savefig(
        pdf,
        metadata={
            "Title": "Drude-bath benchmark",
            "Creator": "Figures/SI/FigS2/plot_FigS2.py",
        },
    )
    fig.savefig(png, dpi=220)
    plt.close(fig)
    return pdf, png


def plot_phonon(root: Path) -> tuple[Path, Path]:
    results = root / "results"
    figures = root
    fig, axes = plt.subplots(
        3, 4, figsize=(13, 8.5), sharex=True, sharey="row"
    )
    letters = "abcdefghijkl"

    for row, cutoff in enumerate(CUTOFFS_MEV):
        for col, lam in enumerate(LAMBDAS_MEV):
            ax = axes[row, col]
            nu, bare = bare_bath_spectrum(lam, cutoff)
            scale = float(np.max(bare))
            bare = bare / scale
            ax.plot(
                nu,
                bare,
                color="black",
                lw=1.7,
                label=r"SCBA: $B_{X,0}(\nu)$",
            )
            ax.axhline(0.0, color="0.7", lw=0.6)
            ax.text(
                0.05,
                0.90,
                f"({letters[row * 4 + col]})",
                transform=ax.transAxes,
                va="top",
                fontsize=24,
            )
            if row == 0:
                ax.set_title(
                    rf"$\lambda={lam:.0f}$ meV", fontsize=15
                )
            if col == 0:
                ax.set_ylabel(
                    rf"$\omega_f={cutoff:.1f}$ meV"
                    + "\nNormalized bath spectrum", fontsize=15
                )
            if row == 2:
                ax.set_xlabel(r"Bath frequency $\nu$ (eV)")
            ax.set_xlim(-0.25, 0.25)
            ax.minorticks_on()
            ax.tick_params(
                which="both", direction="in", top=True, right=True
            )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=1,
        frameon=False,
        fontsize=24,
        bbox_to_anchor=(0.5, 0.985),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    pdf = figures / "FigS2_phonon_diagnostic.pdf"
    png = pdf.with_suffix(".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return pdf, png


def main() -> None:
    configure_si_matplotlib()
    root = FIGURE_ROOT
    outputs = plot_electronic(root)
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
