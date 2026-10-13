#!/usr/bin/env python3
"""Plot the exact-cumulant versus Brownian-SCBA comparisons."""

from __future__ import annotations

from pathlib import Path
from string import ascii_lowercase
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator
import numpy as np

from config import (
    GAMMAS_EV,
    LAMBDAS_MEV,
    STANDARD_FIGURE_GAMMAS_EV,
    gamma_tag,
)

FIGURE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_ROOT.parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "Figures" / "common"))
from si_plot_style import (  # noqa: E402
    BA_STYLE,
    EXACT_STYLE,
    SCBA_STYLE,
    add_panel_label,
    compact_legend_kwargs,
    configure_si_matplotlib,
    style_axis,
)


def load_case(
    root: Path, lambda_mev: float, gamma_eV: float
) -> tuple[np.ndarray, np.ndarray]:
    exact = np.loadtxt(
        root
        / "exact_cumulant"
        / (
            f"cumulant_lambda_{lambda_mev:.0f}_"
            f"gamma_{gamma_tag(gamma_eV)}.txt"
        )
    )
    scba = np.loadtxt(
        root
        / "results"
        / (
            f"scba_lambda_{lambda_mev:.0f}_"
            f"gamma_{gamma_tag(gamma_eV)}.txt"
        )
    )
    return exact, scba


def load_ba_case(
    root: Path, lambda_mev: float, gamma_eV: float
) -> np.ndarray:
    return np.loadtxt(
        root
        / "results"
        / (
            f"single_shot_ba_lambda_{lambda_mev:.0f}_"
            f"gamma_{gamma_tag(gamma_eV)}.txt"
        )
    )


def area_normalize(data: np.ndarray) -> np.ndarray:
    area = float(np.trapz(data[:, 1], data[:, 0]))
    if area <= 0.0:
        raise RuntimeError("cannot area-normalize a nonpositive spectrum")
    return data[:, 1] / area


def decorate(
    fig: plt.Figure,
    axes: np.ndarray,
    *,
    gammas_eV: tuple[float, ...],
    ylabel: str,
) -> None:
    for row, gamma_eV in enumerate(gammas_eV):
        for col, lambda_mev in enumerate(LAMBDAS_MEV):
            ax = axes[row, col]
            ax.set_xlim(1.85, 3.45)
            ax.xaxis.set_major_locator(FixedLocator((2.0, 2.6, 3.2)))
            ax.xaxis.set_minor_locator(MultipleLocator(0.2))
            style_axis(ax)
            ax.tick_params(labelleft=(col == 0))
            add_panel_label(ax, f"({ascii_lowercase[row * 4 + col]})")
            if row == 0:
                ax.set_title(
                    rf"$\lambda = {lambda_mev:.0f}$ meV", pad=3.0
                )
            if col == 0:
                ax.set_ylabel(
                    rf"$\Gamma = {gamma_tag(gamma_eV)}$ eV"
                    + "\n"
                    + ylabel
                )
            if row == len(gammas_eV) - 1:
                ax.set_xlabel("Energy (eV)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(handles),
        bbox_to_anchor=(0.54, 0.997),
        **compact_legend_kwargs(),
    )


def main() -> None:
    configure_si_matplotlib()
    # These spectra are sampled far more finely than the plotted line width.
    # Simplify only the rendered paths (in display coordinates) so the vector
    # PDF does not store millions of visually redundant vertices.  The source
    # arrays and all numerical normalization/integration remain unchanged.
    matplotlib.rcParams.update(
        {
            "path.simplify": True,
            "path.simplify_threshold": 0.05,
        }
    )
    root = FIGURE_ROOT
    figures = root

    fig, axes = plt.subplots(
        3, 4, figsize=(7.08, 5.10), sharex=True, sharey=True
    )
    fig.subplots_adjust(
        left=0.090,
        right=0.992,
        bottom=0.105,
        top=0.885,
        wspace=0.15,
        hspace=0.23,
    )
    for row, gamma_eV in enumerate(STANDARD_FIGURE_GAMMAS_EV):
        for col, lambda_mev in enumerate(LAMBDAS_MEV):
            exact, scba = load_case(root, lambda_mev, gamma_eV)
            axes[row, col].plot(
                exact[:, 0],
                exact[:, 2],
                label="Exact cumulant",
                **EXACT_STYLE,
            )
            axes[row, col].plot(
                scba[:, 0],
                scba[:, 2],
                label="SCBA",
                **SCBA_STYLE,
            )
            axes[row, col].set_ylim(-0.04, 1.09)
    decorate(
        fig,
        axes,
        gammas_eV=STANDARD_FIGURE_GAMMAS_EV,
        ylabel="Norm. intensity",
    )
    for suffix in ("pdf", "png"):
        fig.savefig(
            root / f"FigS3_T=0_Brownian.{suffix}",
            dpi=220 if suffix == "png" else None,
        )
    plt.close(fig)

    fig, axes = plt.subplots(
        4, 4, figsize=(7.08, 6.45), sharex=True, sharey=False
    )
    fig.subplots_adjust(
        left=0.090,
        right=0.992,
        bottom=0.083,
        top=0.900,
        wspace=0.15,
        hspace=0.23,
    )
    for row, gamma_eV in enumerate(GAMMAS_EV):
        for col, lambda_mev in enumerate(LAMBDAS_MEV):
            exact, scba = load_case(root, lambda_mev, gamma_eV)
            ba = load_ba_case(root, lambda_mev, gamma_eV)
            exact_area = area_normalize(exact)
            ba_area = area_normalize(ba)
            scba_area = area_normalize(scba)
            axes[row, col].plot(
                exact[:, 0],
                exact_area,
                label="Exact cumulant",
                **EXACT_STYLE,
            )
            axes[row, col].plot(
                ba[:, 0],
                ba_area,
                label="Single-shot BA",
                **BA_STYLE,
            )
            axes[row, col].plot(
                scba[:, 0],
                scba_area,
                label="SCBA",
                **SCBA_STYLE,
            )
            axes[row, col].set_ylim(0.0, 10.0)
    decorate(
        fig,
        axes,
        gammas_eV=GAMMAS_EV,
        ylabel="Intensity (1/eV)",
    )
    for suffix in ("pdf", "png"):
        save_kwargs = {"dpi": 220} if suffix == "png" else {}
        fig.savefig(
            root / f"FigS3_T=0_Brownian_area_linear.{suffix}",
            **save_kwargs,
        )
        fig.savefig(figures / f"FigS3.{suffix}", **save_kwargs)
    plt.close(fig)

    fig, axes = plt.subplots(
        3, 4, figsize=(7.08, 5.10), sharex=True, sharey=True
    )
    fig.subplots_adjust(
        left=0.090,
        right=0.992,
        bottom=0.105,
        top=0.885,
        wspace=0.15,
        hspace=0.23,
    )
    for row, gamma_eV in enumerate(STANDARD_FIGURE_GAMMAS_EV):
        for col, lambda_mev in enumerate(LAMBDAS_MEV):
            exact, scba = load_case(root, lambda_mev, gamma_eV)
            axes[row, col].semilogy(
                exact[:, 0],
                area_normalize(exact),
                label="Exact cumulant",
                **EXACT_STYLE,
            )
            axes[row, col].semilogy(
                scba[:, 0],
                area_normalize(scba),
                label="SCBA",
                **SCBA_STYLE,
            )
            axes[row, col].set_ylim(1.0e-5, 2.0e3)
    decorate(
        fig,
        axes,
        gammas_eV=STANDARD_FIGURE_GAMMAS_EV,
        ylabel="Intensity (1/eV)",
    )
    for suffix in ("pdf", "png"):
        fig.savefig(
            root / f"FigS3_T=0_Brownian_area_log.{suffix}",
            dpi=220 if suffix == "png" else None,
        )
    plt.close(fig)
    print("Saved Brownian comparison figures")


if __name__ == "__main__":
    main()
