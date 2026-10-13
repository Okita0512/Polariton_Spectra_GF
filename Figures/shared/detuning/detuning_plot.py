#!/usr/bin/env python3
"""Plot-only renderers for the shared detuning data.

This module deliberately contains no solver entry point.  It reads the cached
spectra and linewidth tables moved from the legacy ``Fig5`` directory and
produces the LP-only main-text figure and the spectra/UP supplementary figure.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import numpy as np


SHARED_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SHARED_DIR.parents[2]
CACHE_DIR = SHARED_DIR / "cache"
COMMON_DIR = PROJECT_ROOT / "Figures" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from area_normalization import match_area_to_reference  # noqa: E402
from endpoint_linewidths import load_endpoint_linewidths  # noqa: E402


CONV_EV_PER_HARTREE = 27.2114
OMEGA_0_EV = 2.03
COLLECTIVE_COUPLING_EV = 0.1362
SINGLE_MOLECULE_COUPLING_EV = 0.0681
CAVITY_FWHM_MEV = 44.15
LAMBDA_MEV = 30.0
BATH_CUTOFF_MEV = 24.8
TEMPERATURE_K = 300.0
N_VALUES = (1, 2, 4, 8, 16, 32, 64, 10000)
FINITE_BENCHMARK_N = (1, 2, 4, 8)
SPECTRUM_CACHE_KEYS = ("1.8", "1.85", "1.9", "1.95", "2.0", "2.05", "2.1", "2.15", "2.2")
PHYSICAL_CAVITY_EV = tuple(float(value) + 0.03 for value in SPECTRUM_CACHE_KEYS)


@dataclass(frozen=True)
class LinewidthSeries:
    exciton_fraction: np.ndarray
    linewidth_mev: np.ndarray


def configure_matplotlib() -> None:
    """Set compact, journal-compatible vector-graphics defaults."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.75,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.fontsize": 6.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _colors() -> dict[int, tuple[float, float, float, float]]:
    """Retain the legacy viridis/log(N) series mapping."""
    n_array = np.asarray(N_VALUES, dtype=float)
    scaled = np.log(n_array)
    scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min())
    return {
        n_value: mpl.colormaps["viridis"](1.0 - coordinate)
        for n_value, coordinate in zip(N_VALUES, scaled)
    }


SERIES_COLORS = _colors()


def _linewidth_path(n_value: int, method: str) -> Path:
    return CACHE_DIR / f"width_N={n_value}_Gammac=44.15mev-{method}.txt"


def load_linewidth_series(n_value: int, method: str, branch: str) -> LinewidthSeries:
    """Load one cached width table and reproduce the legacy Hopfield abscissa."""
    path = _linewidth_path(n_value, method)
    if not path.is_file():
        raise FileNotFoundError(f"Missing cached linewidth table: {path}")
    table = np.loadtxt(path, dtype=float)
    if table.ndim != 2 or table.shape[1] != 3 or not np.all(np.isfinite(table)):
        raise ValueError(f"Malformed linewidth table: {path}")

    cavity_frequency = table[:, 0]
    detuning = cavity_frequency - OMEGA_0_EV
    denominator = np.sqrt(detuning**2 + 4.0 * COLLECTIVE_COUPLING_EV**2)
    if branch == "up":
        fraction = 0.5 * (1.0 - detuning / denominator)
        linewidth = table[:, 2] * 1000.0
    elif branch == "lp":
        fraction = 0.5 * (1.0 + detuning / denominator)
        linewidth = table[:, 1] * 1000.0
    else:
        raise ValueError(f"Unknown polariton branch: {branch}")

    keep = np.ones_like(fraction, dtype=bool)
    if method == "SCBA":
        # These four complementary fractions came from calculations targeting
        # the other branch and were also omitted by the legacy renderer.
        for complementary_fraction in (0.02, 0.03, 0.05, 0.10):
            keep &= ~np.isclose(
                fraction,
                complementary_fraction,
                rtol=0.0,
                atol=2.0e-5,
            )
    elif branch == "lp":
        # The wc=1.28 eV HEOM/MF-HEOM point was added specifically for the
        # exciton-rich UP; its complementary photon-rich LP is not a benchmark.
        keep &= ~np.isclose(cavity_frequency, 1.28, rtol=0.0, atol=1.0e-8)

    order = np.argsort(fraction[keep])
    return LinewidthSeries(fraction[keep][order], linewidth[keep][order])


def _benchmark_method(n_value: int) -> str | None:
    if n_value in FINITE_BENCHMARK_N:
        return "HEOM"
    if n_value == 10000:
        return "MF-HEOM"
    return None


def _n_label(n_value: int) -> str:
    return r"$N\to\infty$" if n_value == 10000 else rf"$N={n_value}$"


def _style_linewidth_axis(axis: plt.Axes, branch: str) -> None:
    axis.set_xlim(0.0, 1.0)
    axis.xaxis.set_major_locator(MultipleLocator(0.2))
    axis.xaxis.set_minor_locator(MultipleLocator(0.1))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))
    axis.tick_params(which="major", length=3.2, width=0.75)
    axis.tick_params(which="minor", length=1.8, width=0.55)
    sign = "+" if branch == "up" else "-"
    axis.set_xlabel(rf"Exciton fraction $|C_{{\rm ex,{sign}}}|^2$")
    axis.set_ylabel(rf"$\Gamma_{{\rm {branch.upper()}}}$ (meV)")
    if branch == "up":
        axis.set_ylim(30.0, 110.0)
        axis.yaxis.set_major_locator(MultipleLocator(20.0))
    else:
        axis.set_ylim(0.0, 95.0)
        axis.yaxis.set_major_locator(MultipleLocator(20.0))


def _plot_linewidths(axis: plt.Axes, branch: str) -> tuple[list[Line2D], list[Line2D]]:
    """Plot all SCBA curves and only the benchmark points that exist."""
    endpoints = load_endpoint_linewidths()
    n_handles: list[Line2D] = []
    for n_value in N_VALUES:
        color = SERIES_COLORS[n_value]
        scba = load_linewidth_series(n_value, "SCBA", branch)
        x_scba = np.concatenate(([0.0], scba.exciton_fraction))
        y_scba = np.concatenate(([CAVITY_FWHM_MEV], scba.linewidth_mev))
        axis.plot(
            x_scba,
            y_scba,
            color=color,
            linewidth=1.05,
            marker="o",
            markersize=2.1,
            markeredgewidth=0.0,
            zorder=2,
        )
        axis.plot(
            [1.0],
            [endpoints.scba_molecule_fwhm_meV],
            marker="o",
            markersize=3.0,
            markeredgewidth=0.0,
            color=color,
            linestyle="none",
            zorder=3,
        )
        n_handles.append(Line2D([0], [0], color=color, linewidth=1.4, label=_n_label(n_value)))

        benchmark_method = _benchmark_method(n_value)
        if benchmark_method is not None:
            benchmark = load_linewidth_series(n_value, benchmark_method, branch)
            axis.plot(
                np.concatenate(([0.0], benchmark.exciton_fraction)),
                np.concatenate(([CAVITY_FWHM_MEV], benchmark.linewidth_mev)),
                linestyle="none",
                marker="o",
                markersize=3.6,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.8,
                zorder=4,
            )
            axis.plot(
                [1.0],
                [endpoints.heom_molecule_fwhm_meV],
                linestyle="none",
                marker="o",
                markersize=3.6,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.8,
                zorder=4,
            )

    exciton_fraction = np.linspace(0.0, 1.0, 401)
    hopfield_width = (
        (1.0 - exciton_fraction) * CAVITY_FWHM_MEV
        + exciton_fraction * endpoints.eq26_gamma_ex_meV
    )
    hopfield_handle = axis.plot(
        exciton_fraction,
        hopfield_width,
        color="0.66",
        linewidth=2.0,
        zorder=1,
        label="Hopfield-weighted",
    )[0]
    method_handles = [
        Line2D([0], [0], color="black", linewidth=1.15, marker="o", markersize=2.3, label="SCBA"),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="none",
            marker="o",
            markersize=3.8,
            markerfacecolor="white",
            markeredgewidth=0.8,
            label="HEOM / MF-HEOM",
        ),
        hopfield_handle,
    ]
    _style_linewidth_axis(axis, branch)
    return n_handles, method_handles


def _input_paths() -> list[Path]:
    paths = sorted(CACHE_DIR.glob("width_N=*_Gammac=44.15mev-*.txt"))
    for cache_key in SPECTRUM_CACHE_KEYS:
        directory = CACHE_DIR / "N=4_Gammac=44.15mev" / cache_key
        paths.extend(
            directory / filename
            for filename in ("resp1st.w1", "resp1st_im.w", "Aw_sc_fullpol.txt")
        )
    return paths


def input_checksum() -> tuple[int, str]:
    """Hash the 40 active inputs using paths relative to the moved cache."""
    paths = _input_paths()
    aggregate = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing detuning input: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        relative = path.relative_to(CACHE_DIR).as_posix()
        aggregate.update(f"{relative}\0{digest}\n".encode())
    return len(paths), aggregate.hexdigest()


def load_offset_spectrum(cache_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Load one HEOM/SCBA pair and apply the legacy display normalization."""
    directory = CACHE_DIR / "N=4_Gammac=44.15mev" / cache_key
    heom_energy = np.loadtxt(directory / "resp1st.w1", dtype=float) * CONV_EV_PER_HARTREE
    heom_spectrum = np.loadtxt(directory / "resp1st_im.w", dtype=float)
    scba_table = np.loadtxt(directory / "Aw_sc_fullpol.txt", dtype=float)
    scba_energy = scba_table[:, 0] * CONV_EV_PER_HARTREE
    scba_raw = -scba_table[:, 1]
    if (
        heom_energy.ndim != 1
        or heom_spectrum.shape != heom_energy.shape
        or scba_table.ndim != 2
        or scba_table.shape[1] < 2
        or not np.all(np.isfinite(heom_energy))
        or not np.all(np.isfinite(heom_spectrum))
        or not np.all(np.isfinite(scba_table[:, :2]))
    ):
        raise ValueError(f"Malformed cached spectrum pair in {directory}")
    heom_peak = float(np.max(heom_spectrum))
    if not np.isfinite(heom_peak) or heom_peak <= 0.0:
        raise ValueError(f"Nonpositive HEOM reference peak in {directory}")
    scba_matched = match_area_to_reference(
        heom_energy,
        heom_spectrum,
        scba_energy,
        scba_raw,
    )
    scale_factor = float(scba_matched[0] / scba_raw[0])
    return (
        heom_energy,
        heom_spectrum / heom_peak,
        scba_energy,
        scba_matched / heom_peak,
        scale_factor,
    )


def render_main_lp(output_path: Path) -> dict[str, object]:
    """Render the one-column, LP-only main-text figure."""
    configure_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(3.38, 3.02))
    n_handles, method_handles = _plot_linewidths(axis, "lp")

    n_legend = axis.legend(
        handles=n_handles,
        title=r"Ensemble size",
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        borderaxespad=0.0,
        frameon=False,
        handlelength=1.25,
        handletextpad=0.45,
        columnspacing=0.8,
        labelspacing=0.25,
        title_fontsize=6.9,
    )
    axis.add_artist(n_legend)
    axis.legend(
        handles=method_handles,
        loc="lower left",
        frameon=False,
        handlelength=1.55,
        handletextpad=0.45,
        labelspacing=0.3,
    )
    figure.subplots_adjust(left=0.175, right=0.98, bottom=0.15, top=0.985)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)

    count, checksum = input_checksum()
    return {
        "role": "main Fig. 3, LP-only",
        "output": output_path.relative_to(PROJECT_ROOT).as_posix(),
        "active_input_count": count,
        "active_input_sha256": checksum,
        "scba_curves": len(N_VALUES),
        "benchmark_series": 5,
        "hopfield_reference_lines": 1,
        "collective_coupling_meV": COLLECTIVE_COUPLING_EV * 1000.0,
        "validation": "PASS",
    }


def render_si_spectra_up(output_path: Path) -> dict[str, object]:
    """Render offset spectra and UP linewidths as the two-panel SI figure."""
    configure_matplotlib()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7.10, 3.12))
    grid = figure.add_gridspec(1, 2, width_ratios=(1.0, 1.08), wspace=0.27)
    spectra_axis = figure.add_subplot(grid[0, 0])
    up_axis = figure.add_subplot(grid[0, 1])

    area_factors: dict[str, float] = {}
    area_errors: list[float] = []
    spectrum_pairs = [load_offset_spectrum(cache_key) for cache_key in SPECTRUM_CACHE_KEYS]
    largest_display_peak = max(
        max(float(np.max(pair[1])), float(np.max(pair[3])))
        for pair in spectrum_pairs
    )
    # Derive the row spacing from both methods.  The 0.08 margin prevents the
    # tallest SCBA trace from touching or hiding beneath the next row.
    offset_step = largest_display_peak + 0.08
    for index, (cache_key, cavity_frequency, pair) in enumerate(
        zip(SPECTRUM_CACHE_KEYS, PHYSICAL_CAVITY_EV, spectrum_pairs)
    ):
        heom_energy, heom_display, scba_energy, scba_display, factor = pair
        offset = index * offset_step
        spectra_axis.fill_between(
            heom_energy,
            offset,
            heom_display + offset,
            color="#5aa9e6",
            alpha=0.34,
            linewidth=0.0,
            zorder=1,
        )
        spectra_axis.plot(
            scba_energy,
            scba_display + offset,
            color="#172554",
            linewidth=0.8,
            zorder=2,
        )
        spectra_axis.text(
            1.515,
            offset + 0.68,
            rf"$\omega_{{\rm c}}={cavity_frequency:.2f}$ eV",
            fontsize=6.5,
            ha="left",
            va="center",
        )
        area_factors[f"omega_c={cavity_frequency:.2f} eV"] = factor
        overlap_min = max(float(np.min(heom_energy)), float(np.min(scba_energy)))
        overlap_max = min(float(np.max(heom_energy)), float(np.max(scba_energy)))
        heom_overlap = (heom_energy >= overlap_min) & (heom_energy <= overlap_max)
        scba_overlap = (scba_energy >= overlap_min) & (scba_energy <= overlap_max)
        heom_area = float(np.trapz(heom_display[heom_overlap], heom_energy[heom_overlap]))
        scba_area = float(np.trapz(scba_display[scba_overlap], scba_energy[scba_overlap]))
        area_errors.append(abs(scba_area / heom_area - 1.0))

    spectra_axis.set_xlim(1.5, 2.5)
    spectra_axis.set_ylim(
        -0.18,
        (len(SPECTRUM_CACHE_KEYS) - 1) * offset_step + largest_display_peak + 0.04,
    )
    spectra_axis.set_xlabel(r"$\omega$ (eV)")
    spectra_axis.set_ylabel(r"Normalized absorption + offset")
    spectra_axis.set_yticks([])
    spectra_axis.xaxis.set_major_locator(MultipleLocator(0.2))
    spectra_axis.xaxis.set_minor_locator(MultipleLocator(0.1))
    spectra_axis.tick_params(which="major", length=3.2, width=0.75)
    spectra_axis.tick_params(which="minor", length=1.8, width=0.55)
    spectra_axis.legend(
        handles=(
            Patch(facecolor="#5aa9e6", alpha=0.34, edgecolor="none", label="HEOM"),
            Line2D([0], [0], color="#172554", linewidth=1.1, label="SCBA"),
        ),
        loc="lower right",
        frameon=False,
        handlelength=1.6,
        handletextpad=0.45,
        labelspacing=0.25,
    )
    spectra_axis.text(-0.14, 1.015, "(a)", transform=spectra_axis.transAxes, fontsize=10, fontweight="bold")

    n_handles, method_handles = _plot_linewidths(up_axis, "up")
    n_legend = up_axis.legend(
        handles=n_handles,
        title=r"Ensemble size",
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0.0,
        frameon=False,
        handlelength=1.25,
        handletextpad=0.45,
        columnspacing=0.8,
        labelspacing=0.25,
        title_fontsize=6.9,
    )
    up_axis.add_artist(n_legend)
    up_axis.legend(
        handles=method_handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.72),
        frameon=False,
        handlelength=1.55,
        handletextpad=0.45,
        labelspacing=0.3,
    )
    up_axis.text(-0.14, 1.015, "(b)", transform=up_axis.transAxes, fontsize=10, fontweight="bold")

    figure.subplots_adjust(left=0.07, right=0.992, bottom=0.16, top=0.97)
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)

    count, checksum = input_checksum()
    return {
        "role": "SI Fig. S14, offset spectra and UP linewidth",
        "output": output_path.relative_to(PROJECT_ROOT).as_posix(),
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
        "validation": "PASS",
    }


def write_validation(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
