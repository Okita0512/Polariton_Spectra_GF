#!/usr/bin/env python3
"""Plot the eight publication figure families for the single-mode benchmark.

The script consumes only the completed combined NPZ archives and the
machine-readable CSV summaries in ``results/``.  It selects cases by physical
metadata rather than filename spelling and never depends on the process
working directory.  Every figure is saved as both PDF and PNG below the local
``figures/`` directory unless an explicit output directory is supplied.

All spectral panels show the *bright optical absorption*.  Dark projected
spectra and the site-averaged local DOS are not plotted or described as
absorption here.  Spectral overlays are explicitly normalized to unit area on
the declared analysis window only after the raw areas have been retained in
the analysis CSV.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS = MODULE_DIR / "results"
DEFAULT_FIGURES = MODULE_DIR / "figures"

FIGURE_NAMES = (
    "direct_analogue_spectra",
    "j0_coupling_progression",
    "error_heatmaps",
    "signed_j_error_curves",
    "eta_convergence",
    "self_energy_channels",
    "ed_truncation_convergence",
    "regime_map",
)

ED_COLOR = "#202020"
BA_COLOR = "#D55E00"
SCBA_COLOR = "#0072B2"
TEMPERATURE_COLORS = {0.0: "#3B4CC0", 300.0: "#B40426"}
S_VALUES = (0.05, 0.10, 0.25, 0.50, 1.0, 2.0)
R_VALUES = (-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class ArchiveData:
    energy_eV: np.ndarray
    analysis_mask: np.ndarray
    ed_bright: np.ndarray
    ba_bright: np.ndarray
    scba_bright: np.ndarray
    scba_sigma: np.ndarray
    channels: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError("Required table is missing: %s" % path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header: %s" % path)
        return [dict(row) for row in reader]


def _float(row: Dict[str, str], key: str) -> float:
    text = row.get(key, "")
    if text is None or text.strip() == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _bool(row: Dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() in ("true", "1", "yes")


def _families(row: Dict[str, str]) -> Tuple[str, ...]:
    return tuple(
        item for item in row.get("manifest_families", "").split(";") if item
    )


def _close(value: float, target: float, tolerance: float = 1.0e-8) -> bool:
    return bool(np.isfinite(value) and abs(value - target) <= tolerance)


def _select(
    rows: Sequence[Dict[str, str]],
    family: Optional[str] = None,
    **conditions: float,
) -> List[Dict[str, str]]:
    selected = []
    for row in rows:
        if family is not None and family not in _families(row):
            continue
        if all(_close(_float(row, key), value) for key, value in conditions.items()):
            selected.append(row)
    return selected


def _one(
    rows: Sequence[Dict[str, str]],
    family: Optional[str] = None,
    **conditions: float,
) -> Dict[str, str]:
    selected = _select(rows, family=family, **conditions)
    if len(selected) != 1:
        raise ValueError(
            "Expected one case for family=%r conditions=%r; found %d"
            % (family, conditions, len(selected))
        )
    return selected[0]


def _resolve_archive(row: Dict[str, str], results_dir: Path) -> Path:
    candidates = []
    text = row.get("combined_archive", "").strip()
    if text:
        candidates.append(Path(text))
    output = row.get("manifest_output_file", "").strip()
    if output:
        candidates.append(MODULE_DIR / output)
        candidates.append(results_dir.parent / output)
    candidates.append(results_dir / "spectra" / (row["case_id"] + ".npz"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "No combined archive found for %s; tried %s"
        % (row.get("case_id", "unknown"), candidates)
    )


def _load_archive(row: Dict[str, str], results_dir: Path) -> ArchiveData:
    path = _resolve_archive(row, results_dir)
    required = (
        "energy_eV",
        "analysis_mask",
        "ed_green_bright_per_eV",
        "ba_green_bright_per_eV",
        "scba_green_bright_per_eV",
        "scba_sigma_eV",
        "scba_sigma_from_bright_emission_eV",
        "scba_sigma_from_bright_absorption_eV",
        "scba_sigma_from_dark_emission_eV",
        "scba_sigma_from_dark_absorption_eV",
        "metadata_json",
    )
    with np.load(str(path), allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise KeyError("Archive %s lacks %s" % (path, missing))
        energy = np.asarray(archive["energy_eV"], dtype=float).copy()
        mask = np.asarray(archive["analysis_mask"], dtype=bool).copy()
        ed = np.asarray(archive["ed_green_bright_per_eV"], dtype=complex).copy()
        ba = np.asarray(archive["ba_green_bright_per_eV"], dtype=complex).copy()
        scba = np.asarray(archive["scba_green_bright_per_eV"], dtype=complex).copy()
        sigma = np.asarray(archive["scba_sigma_eV"], dtype=complex).copy()
        channels = {
            name: np.asarray(
                archive["scba_sigma_from_%s_eV" % name], dtype=complex
            ).copy()
            for name in (
                "bright_emission",
                "bright_absorption",
                "dark_emission",
                "dark_absorption",
            )
        }
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    arrays = (ed, ba, scba, sigma) + tuple(channels.values())
    if mask.shape != energy.shape or any(array.shape != energy.shape for array in arrays):
        raise ValueError("Inconsistent array shapes in %s" % path)
    if not all(np.all(np.isfinite(array)) for array in (energy,) + arrays):
        raise ValueError("Non-finite numerical data in %s" % path)
    if metadata.get("case_id") != row.get("case_id"):
        raise ValueError("Archive/summary case-ID mismatch for %s" % path)
    return ArchiveData(energy, mask, ed, ba, scba, sigma, channels, metadata)


def _unit_area_spectrum(green: np.ndarray, energy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    spectrum = -np.imag(green)
    area = float(np.trapz(spectrum[mask], energy[mask]))
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("Cannot normalize a nonpositive spectral area")
    return spectrum / area


def _save_figure(fig: plt.Figure, output_dir: Path, name: str, dpi: int) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("pdf", "png"):
        path = output_dir / (name + "." + suffix)
        fig.savefig(
            str(path),
            dpi=dpi if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.04,
        )
        paths.append(path.resolve())
    plt.close(fig)
    return paths


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        0.015,
        0.975,
        label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )


def plot_coupled_s_spectra(
    rows: Sequence[Dict[str, str]],
    results_dir: Path,
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    """Plot the intrinsic T=0, resonantly coupled dimer as S is varied.

    The archived omega0=50 meV cases are displayed in dimensionless units,
    so no Drude-bath parameter matching enters the comparison.  Fixing
    r=2J/omega0=+1 puts the bright upper exciton at the one-phonon emission
    resonance with the dark exciton; only S changes between panels.
    """

    selected_s = (0.5, 1.0, 2.0)
    omega0_meV = 50.0
    omega0_eV = omega0_meV / 1000.0
    ratio = 1.0
    eta_meV = 0.5
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.75), sharex=True, sharey=True)
    handles = None
    for panel, (axis, huang_rhys) in enumerate(zip(axes, selected_s)):
        row = _one(
            rows,
            family="heatmap_r_S",
            omega0_meV=omega0_meV,
            huang_rhys=huang_rhys,
            r_2J_over_omega0=ratio,
            temperature_K=0.0,
            eta_meV=eta_meV,
        )
        data = _load_archive(row, results_dir)
        bare_bright_eV = _float(row, "bare_bright_energy_eV")
        detuning = (data.energy_eV - bare_bright_eV) / omega0_eV
        display = (detuning >= -3.5) & (detuning <= 6.0)
        curves = (
            _unit_area_spectrum(data.ed_bright, data.energy_eV, data.analysis_mask),
            _unit_area_spectrum(data.ba_bright, data.energy_eV, data.analysis_mask),
            _unit_area_spectrum(data.scba_bright, data.energy_eV, data.analysis_mask),
        )
        # Multiplication by omega0 converts the unit-area density from eV^-1
        # to the dimensionless detuning coordinate used on the horizontal axis.
        dimensionless_curves = tuple(omega0_eV * curve for curve in curves)
        line_ed, = axis.plot(detuning[display], dimensionless_curves[0][display], color=ED_COLOR, label="ED", zorder=4)
        line_ba, = axis.plot(detuning[display], dimensionless_curves[1][display], color=BA_COLOR, label="BA", alpha=0.90)
        line_scba, = axis.plot(detuning[display], dimensionless_curves[2][display], color=SCBA_COLOR, label="SCBA", alpha=0.95)
        handles = (line_ed, line_ba, line_scba)
        axis.set_title(r"$S=%g$" % huang_rhys)
        axis.set_xlim(-3.5, 6.0)
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.16, linewidth=0.5)
        axis.text(
            0.975,
            0.95,
            "SCBA vs ED\n$L_1=%.3f$; $W=%.2f$ meV\npeaks ED/SCBA: $%d/%d$"
            % (
                _float(row, "scba_unit_area_l1"),
                1000.0 * _float(row, "scba_wasserstein_eV"),
                int(round(_float(row, "ed_resolved_peak_count"))),
                int(round(_float(row, "scba_resolved_peak_count"))),
            ),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.8),
        )
        _panel_label(axis, "(%s)" % chr(ord("a") + panel))
    if handles is not None:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.905),
            ncol=3,
            frameon=False,
        )
    fig.supxlabel(r"Detuning from the bare bright exciton, $(E-E_B)/\omega_0$")
    fig.supylabel(r"Unit-area bright absorption, $\omega_0 A_B(E)$")
    fig.suptitle(
        r"Resonantly coupled single-mode Frenkel dimer at $T=0$: "
        r"$2J/\omega_0=1$, $\eta/\omega_0=0.01$",
        y=0.995,
    )
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.78, wspace=0.10)
    return _save_figure(fig, output_dir, "direct_analogue_spectra", dpi)


def plot_j0_progression(
    rows: Sequence[Dict[str, str]],
    results_dir: Path,
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    selected_s = (0.05, 0.50, 2.0)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.55), sharex=True)
    handles = None
    for panel, (axis, huang_rhys) in enumerate(zip(axes, selected_s)):
        row = _one(
            rows,
            omega0_meV=50.0,
            huang_rhys=huang_rhys,
            j_meV=0.0,
            temperature_K=0.0,
            eta_meV=0.5,
        )
        data = _load_archive(row, results_dir)
        display = (data.energy_eV >= 1.94) & (data.energy_eV <= 2.42)
        curves = (
            _unit_area_spectrum(data.ed_bright, data.energy_eV, data.analysis_mask),
            _unit_area_spectrum(data.ba_bright, data.energy_eV, data.analysis_mask),
            _unit_area_spectrum(data.scba_bright, data.energy_eV, data.analysis_mask),
        )
        line_ed, = axis.plot(data.energy_eV[display], curves[0][display], color=ED_COLOR, label="ED", zorder=4)
        line_ba, = axis.plot(data.energy_eV[display], curves[1][display], color=BA_COLOR, label="BA")
        line_scba, = axis.plot(data.energy_eV[display], curves[2][display], color=SCBA_COLOR, label="SCBA")
        handles = (line_ed, line_ba, line_scba)
        axis.set_title(r"$S=%g$" % huang_rhys)
        axis.set_xlim(1.94, 2.42)
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.16, linewidth=0.5)
        _panel_label(axis, "(%s)" % chr(ord("a") + panel))
    if handles is not None:
        axes[1].legend(handles=handles, loc="upper right", frameon=False)
    axes[0].set_ylabel(r"Unit-area bright absorption (eV$^{-1}$)")
    for axis in axes:
        axis.set_xlabel("Photon energy (eV)")
    fig.suptitle(
        r"Local Franck--Condon benchmark at $J=0$, $T=0$, "
        r"$\omega_0=50$ meV, $\eta=0.5$ meV",
        y=1.01,
    )
    fig.subplots_adjust(wspace=0.18)
    return _save_figure(fig, output_dir, "j0_coupling_progression", dpi)


def _heatmap_matrix(
    rows: Sequence[Dict[str, str]],
    temperature_K: float,
    key: str,
    reliable_key: Optional[str] = None,
    scale: float = 1.0,
) -> np.ndarray:
    matrix = np.full((len(S_VALUES), len(R_VALUES)), np.nan)
    for i, huang_rhys in enumerate(S_VALUES):
        for j, ratio in enumerate(R_VALUES):
            row = _one(
                rows,
                family="heatmap_r_S",
                huang_rhys=huang_rhys,
                r_2J_over_omega0=ratio,
                temperature_K=temperature_K,
                omega0_meV=50.0,
                eta_meV=0.5,
            )
            if reliable_key is not None and not _bool(row, reliable_key):
                continue
            value = _float(row, key)
            if np.isfinite(value):
                matrix[i, j] = scale * abs(value)
    return matrix


def plot_error_heatmaps(
    rows: Sequence[Dict[str, str]],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    metrics = (
        ("scba_unit_area_l1", None, 1.0, r"Unit-area $L_1$"),
        ("scba_wasserstein_eV", None, 1.0e3, "Wasserstein (meV)"),
        (
            "scba_sideband_aggregate_relative_weight_error",
            None,
            1.0,
            "Sideband weight error",
        ),
        (
            "scba_peak_comparison_dominant_peak_shift_candidate_minus_reference_eV",
            "scba_peak_comparison_dominant_peak_shift_available",
            1.0e3,
            r"$|$Dominant peak shift$|$ (meV)",
        ),
        (
            "scba_peak_comparison_fwhm_error_candidate_minus_reference_eV",
            "scba_peak_comparison_fwhm_error_reliable",
            1.0e3,
            r"$|\Delta$FWHM$|$ (meV)",
        ),
    )
    matrices = {
        (temperature, column): _heatmap_matrix(
            rows,
            temperature,
            key,
            reliable,
            scale,
        )
        for temperature in (0.0, 300.0)
        for column, (key, reliable, scale, _) in enumerate(metrics)
    }
    fig = plt.figure(figsize=(19.2, 6.2))
    grid = fig.add_gridspec(
        2,
        10,
        width_ratios=(1.0, 0.045) * 5,
        wspace=0.42,
        hspace=0.18,
    )
    axes = np.empty((2, 5), dtype=object)
    colorbar_axes = []
    for row_index in range(2):
        for column in range(5):
            axes[row_index, column] = fig.add_subplot(grid[row_index, 2 * column])
    for column in range(5):
        colorbar_axes.append(fig.add_subplot(grid[:, 2 * column + 1]))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#D9D9D9")
    for column, (_, reliable, _, title) in enumerate(metrics):
        finite_values = np.concatenate(
            [
                matrix[np.isfinite(matrix)]
                for matrix in (matrices[(0.0, column)], matrices[(300.0, column)])
                if np.any(np.isfinite(matrix))
            ]
        )
        maximum = float(np.max(finite_values)) if finite_values.size else 1.0
        norm = Normalize(vmin=0.0, vmax=maximum if maximum > 0.0 else 1.0)
        images = []
        for row_index, temperature in enumerate((0.0, 300.0)):
            axis = axes[row_index, column]
            matrix = matrices[(temperature, column)]
            image = axis.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm)
            images.append(image)
            if reliable is not None:
                for i, j in zip(*np.where(~np.isfinite(matrix))):
                    axis.text(j, i, "N/A", ha="center", va="center", fontsize=5.8, color="#555555")
            axis.set_xticks(range(len(R_VALUES)))
            axis.set_xticklabels(["%g" % value for value in R_VALUES], rotation=0)
            axis.set_yticks(range(len(S_VALUES)))
            if column == 0:
                axis.set_yticklabels(["%g" % value for value in S_VALUES])
            else:
                axis.tick_params(axis="y", labelleft=False)
            if row_index == 0:
                axis.set_title(title)
            if column == 0:
                axis.set_ylabel(r"$S$\n$T=%g$ K" % temperature)
            if row_index == 1:
                axis.set_xlabel(r"$r=2J/\omega_0$")
            axis.text(
                -0.04,
                1.02,
                "(%s)" % chr(ord("a") + row_index * 5 + column),
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        colorbar = fig.colorbar(images[0], cax=colorbar_axes[column])
        colorbar.ax.tick_params(labelsize=7.5)
    fig.suptitle(
        r"SCBA error relative to ED: $\omega_0=50$ meV, $\eta=0.5$ meV; "
        "gray N/A cells lack an available peak or reliable single-peak FWHM",
        y=1.015,
    )
    return _save_figure(fig, output_dir, "error_heatmaps", dpi)


def plot_signed_j_curves(
    rows: Sequence[Dict[str, str]],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.6), sharex=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(S_VALUES)))
    for column, temperature in enumerate((0.0, 300.0)):
        for huang_rhys, color in zip(S_VALUES, colors):
            selected = _select(
                rows,
                family="heatmap_r_S",
                huang_rhys=huang_rhys,
                temperature_K=temperature,
                omega0_meV=50.0,
                eta_meV=0.5,
            )
            selected.sort(key=lambda row: _float(row, "r_2J_over_omega0"))
            ratio = np.asarray([_float(row, "r_2J_over_omega0") for row in selected])
            l1 = np.asarray([_float(row, "scba_unit_area_l1") for row in selected])
            wasserstein = 1.0e3 * np.asarray(
                [_float(row, "scba_wasserstein_eV") for row in selected]
            )
            axes[0, column].plot(ratio, l1, marker="o", markersize=3.2, color=color, label=r"$S=%g$" % huang_rhys)
            axes[1, column].plot(ratio, wasserstein, marker="o", markersize=3.2, color=color)
        axes[0, column].set_title(r"$T=%g$ K" % temperature)
        for axis in axes[:, column]:
            axis.axvline(-1.0, color="#777777", linestyle=":", linewidth=0.8)
            axis.axvline(+1.0, color="#777777", linestyle=":", linewidth=0.8)
            axis.axvline(0.0, color="#999999", linestyle="--", linewidth=0.65)
            axis.grid(alpha=0.16, linewidth=0.5)
            axis.set_xlim(-4.15, 4.15)
    axes[0, 0].set_ylabel(r"Unit-area $L_1$")
    axes[1, 0].set_ylabel("Wasserstein distance (meV)")
    axes[1, 0].set_xlabel(r"$r=2J/\omega_0$")
    axes[1, 1].set_xlabel(r"$r=2J/\omega_0$")
    axes[0, 1].legend(frameon=False, ncol=2, loc="upper right")
    for index, axis in enumerate(axes.ravel()):
        _panel_label(axis, "(%s)" % chr(ord("a") + index))
    fig.suptitle(
        "Signed-coupling SCBA error: dotted lines mark one-phonon bright--dark resonance",
        y=1.01,
    )
    fig.subplots_adjust(wspace=0.20, hspace=0.18)
    return _save_figure(fig, output_dir, "signed_j_error_curves", dpi)


def plot_eta_convergence(
    rows: Sequence[Dict[str, str]],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    regimes = (
        ("weak, off resonant", dict(omega0_meV=100.0, huang_rhys=0.10, r_2J_over_omega0=4.0, temperature_K=0.0)),
        ("local FC", dict(omega0_meV=100.0, huang_rhys=1.0, r_2J_over_omega0=0.0, temperature_K=0.0)),
        ("emission resonance", dict(omega0_meV=100.0, huang_rhys=1.0, r_2J_over_omega0=1.0, temperature_K=0.0)),
        ("lower bright", dict(omega0_meV=100.0, huang_rhys=1.0, r_2J_over_omega0=-1.0, temperature_K=0.0)),
        ("direct $J=-12.4$", dict(omega0_meV=24.8, huang_rhys=30.0 / 24.8, j_meV=-12.4, temperature_K=300.0)),
        ("direct $J=0$", dict(omega0_meV=24.8, huang_rhys=30.0 / 24.8, j_meV=0.0, temperature_K=300.0)),
        ("direct $J=+12.4$", dict(omega0_meV=24.8, huang_rhys=30.0 / 24.8, j_meV=12.4, temperature_K=300.0)),
    )
    colors = plt.get_cmap("tab10")(np.arange(len(regimes)))
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharex=True)
    for (label, conditions), color in zip(regimes, colors):
        selected = _select(rows, **conditions)
        selected = [
            row
            for row in selected
            if any(
                _close(_float(row, "eta_meV"), target)
                for target in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
            )
        ]
        selected.sort(key=lambda row: _float(row, "eta_meV"))
        eta = np.asarray([_float(row, "eta_meV") for row in selected])
        l1 = np.asarray([_float(row, "scba_unit_area_l1") for row in selected])
        wasserstein = 1.0e3 * np.asarray([_float(row, "scba_wasserstein_eV") for row in selected])
        if eta.size < 5:
            raise ValueError("Eta series %s has only %d points" % (label, eta.size))
        linestyle = "--" if label.startswith("direct") else "-"
        axes[0].plot(eta, l1, marker="o", markersize=3.5, color=color, linestyle=linestyle, label=label)
        axes[1].plot(eta, wasserstein, marker="o", markersize=3.5, color=color, linestyle=linestyle)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel(r"Common Lorentz HWHM $\eta$ (meV)")
        axis.grid(alpha=0.18, which="both", linewidth=0.5)
    axes[0].set_ylabel(r"SCBA unit-area $L_1$")
    axes[1].set_ylabel("SCBA Wasserstein distance (meV)")
    axes[0].legend(frameon=False, fontsize=7.2, ncol=2, loc="best")
    _panel_label(axes[0], "(a)")
    _panel_label(axes[1], "(b)")
    fig.suptitle(
        r"Resolution convergence: $\eta$ broadens spectra but does not damp the oscillator",
        y=1.01,
    )
    fig.subplots_adjust(wspace=0.22)
    return _save_figure(fig, output_dir, "eta_convergence", dpi)


def plot_self_energy_channels(
    rows: Sequence[Dict[str, str]],
    results_dir: Path,
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    channel_order = (
        "bright_emission",
        "dark_emission",
        "bright_absorption",
        "dark_absorption",
    )
    labels = {
        "bright_emission": "bright internal: phonon emission",
        "dark_emission": "dark internal: phonon emission",
        "bright_absorption": "bright internal: phonon absorption",
        "dark_absorption": "dark internal: phonon absorption",
    }
    colors = {
        "bright_emission": "#0072B2",
        "dark_emission": "#D55E00",
        "bright_absorption": "#56B4E9",
        "dark_absorption": "#CC79A7",
    }
    styles = {
        "bright_emission": "-",
        "dark_emission": "-",
        "bright_absorption": "--",
        "dark_absorption": "--",
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.5), sharex=True)
    for column, temperature in enumerate((0.0, 300.0)):
        selected = _select(
            rows,
            family="heatmap_r_S",
            omega0_meV=50.0,
            huang_rhys=1.0,
            temperature_K=temperature,
            eta_meV=0.5,
        )
        selected.sort(key=lambda row: _float(row, "r_2J_over_omega0"))
        ratios = []
        real_values = {name: [] for name in channel_order}
        linewidth_values = {name: [] for name in channel_order}
        for row in selected:
            data = _load_archive(row, results_dir)
            valid_indices = np.flatnonzero(data.analysis_mask)
            peak_index = int(valid_indices[np.argmax(-np.imag(data.scba_bright[valid_indices]))])
            channel_sum = sum(data.channels[name][peak_index] for name in channel_order)
            closure = abs(channel_sum - data.scba_sigma[peak_index])
            if closure > 2.0e-8:
                raise RuntimeError("Self-energy channel closure failed for %s" % row["case_id"])
            ratios.append(_float(row, "r_2J_over_omega0"))
            for name in channel_order:
                value = data.channels[name][peak_index]
                real_values[name].append(1.0e3 * value.real)
                linewidth_values[name].append(-2.0e3 * value.imag)
        for name in channel_order:
            axes[0, column].plot(ratios, real_values[name], color=colors[name], linestyle=styles[name], marker="o", markersize=3.0, label=labels[name])
            axes[1, column].plot(ratios, linewidth_values[name], color=colors[name], linestyle=styles[name], marker="o", markersize=3.0)
        axes[0, column].set_title(r"$T=%g$ K" % temperature)
        for axis in axes[:, column]:
            axis.axvline(-1.0, color="#888888", linestyle=":", linewidth=0.8)
            axis.axvline(+1.0, color="#888888", linestyle=":", linewidth=0.8)
            axis.axhline(0.0, color="#999999", linewidth=0.6)
            axis.grid(alpha=0.16, linewidth=0.5)
    axes[0, 0].set_ylabel(r"Re $\Sigma_c$ (meV)")
    axes[1, 0].set_ylabel(r"$-2$ Im $\Sigma_c$ (meV)")
    axes[1, 0].set_xlabel(r"$r=2J/\omega_0$")
    axes[1, 1].set_xlabel(r"$r=2J/\omega_0$")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        frameon=False,
        fontsize=7.4,
        ncol=4,
    )
    for index, axis in enumerate(axes.ravel()):
        _panel_label(axis, "(%s)" % chr(ord("a") + index))
    fig.suptitle(
        r"SCBA internal-channel decomposition at its bright absorption maximum "
        r"($S=1$, $\omega_0=50$ meV, $\eta=0.5$ meV)",
        y=1.01,
    )
    fig.subplots_adjust(wspace=0.20, hspace=0.18, top=0.84)
    return _save_figure(fig, output_dir, "self_energy_channels", dpi)


def plot_ed_convergence(
    convergence_rows: Sequence[Dict[str, str]],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    case_order = []
    for row in convergence_rows:
        if row["case_id"] not in case_order:
            case_order.append(row["case_id"])
    if len(case_order) != 3:
        raise ValueError("Expected three ED convergence cases; found %d" % len(case_order))
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.0), sharex="col")
    floor = 1.0e-15
    for column, case_id in enumerate(case_order):
        selected = [row for row in convergence_rows if row["case_id"] == case_id]
        selected.sort(key=lambda row: _float(row, "n_vib"))
        n_vib = np.asarray([_float(row, "n_vib") for row in selected])
        l1 = np.maximum(
            np.asarray([abs(_float(row, "unit_area_l1")) for row in selected]),
            floor,
        )
        missing = np.maximum(
            np.asarray(
                [
                    abs(_float(row, "raw_bright_missing_from_retained_thermal"))
                    for row in selected
                ]
            ),
            floor,
        )
        reference_n = int(round(_float(selected[0], "reference_n_vib")))
        passed = np.asarray([_bool(row, "comparison_to_reference_pass") for row in selected])
        axes[0, column].semilogy(n_vib, l1, color=SCBA_COLOR, marker="o", markersize=4)
        axes[1, column].semilogy(n_vib, missing, color=BA_COLOR, marker="s", markersize=4)
        for axis in axes[:, column]:
            axis.axvline(reference_n, color="#777777", linestyle="--", linewidth=0.8)
            axis.grid(alpha=0.18, which="both", linewidth=0.5)
            axis.set_xticks(n_vib.astype(int))
        for x_value, pass_value in zip(n_vib, passed):
            if pass_value:
                axes[0, column].plot(x_value, floor * 2.0, marker="|", color="#009E73", markersize=7)
        title = (
            r"$S=%g$, $r=%g$, $T=%g$ K"
            % (
                _float(selected[0], "S"),
                _float(selected[0], "r_2J_over_omega0"),
                _float(selected[0], "temperature_K"),
            )
        )
        axes[0, column].set_title(title)
        axes[1, column].set_xlabel(r"$N_{\rm vib}$ states/mode")
        for axis, label in (
            (axes[0, column], "(%s)" % chr(ord("a") + column)),
            (axes[1, column], "(%s)" % chr(ord("d") + column)),
        ):
            axis.text(
                0.015,
                1.015,
                label,
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
    axes[0, 0].set_ylabel(r"Unit-area $L_1$ vs reference")
    axes[1, 0].set_ylabel("Missing retained thermal weight")
    fig.suptitle(
        "ED truncation convergence; dashed line is the reference cutoff "
        r"(values $\leq10^{-15}$ shown at the floor)",
        y=1.01,
    )
    fig.subplots_adjust(wspace=0.22, hspace=0.20)
    return _save_figure(fig, output_dir, "ed_truncation_convergence", dpi)


def _regime_categories(values: np.ndarray) -> np.ndarray:
    boundaries = np.asarray((0.0, 0.25, 0.75, 1.25, np.inf))
    return np.digitize(values, boundaries[1:-1], right=False)


def plot_regime_map(
    rows: Sequence[Dict[str, str]],
    output_dir: Path,
    dpi: int,
) -> List[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5))
    category_colors = ("#2CA25F", "#FDD049", "#F28E2B", "#C43C39")
    category_cmap = ListedColormap(category_colors)
    category_norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5, 3.5), category_cmap.N)
    for column, temperature in enumerate((0.0, 300.0)):
        l1 = _heatmap_matrix(rows, temperature, "scba_unit_area_l1")
        categories = _regime_categories(l1)
        axis = axes[0, column]
        axis.imshow(categories, origin="lower", aspect="auto", cmap=category_cmap, norm=category_norm)
        axis.set_xticks(range(len(R_VALUES)))
        axis.set_xticklabels(["%g" % value for value in R_VALUES])
        axis.set_yticks(range(len(S_VALUES)))
        axis.set_yticklabels(["%g" % value for value in S_VALUES])
        axis.set_xlabel(r"$r=2J/\omega_0$")
        axis.set_ylabel(r"$S$")
        axis.set_title(r"$T=%g$ K" % temperature)
        axis.axvline(3.0, color="white", linewidth=0.6, alpha=0.8)
        axis.axvline(5.0, color="white", linewidth=0.6, alpha=0.8)
        for i in range(len(S_VALUES)):
            for j in range(len(R_VALUES)):
                axis.text(j, i, "%.2f" % l1[i, j], ha="center", va="center", fontsize=5.8, color="black" if categories[i, j] < 2 else "white")
        _panel_label(axis, "(%s)" % chr(ord("a") + column))

    large = [row for row in rows if "large_splitting" in _families(row)]
    if not large:
        raise ValueError("No large_splitting cases are present in the error summary")
    selected_s = (0.10, 1.0, 2.0)
    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(selected_s)))
    for huang_rhys, color in zip(selected_s, colors):
        for sign, linestyle, marker in ((-1, "--", "s"), (+1, "-", "o")):
            selected = [
                row
                for row in large
                if _close(_float(row, "huang_rhys"), huang_rhys)
                and np.sign(_float(row, "r_2J_over_omega0")) == sign
                and _close(_float(row, "temperature_K"), 0.0)
                and _close(_float(row, "omega0_meV"), 50.0)
                and _close(_float(row, "eta_meV"), 0.5)
            ]
            selected.sort(key=lambda row: abs(_float(row, "r_2J_over_omega0")))
            absolute_r = np.asarray([abs(_float(row, "r_2J_over_omega0")) for row in selected])
            scba_l1 = np.asarray([_float(row, "scba_unit_area_l1") for row in selected])
            common_wasserstein_meV = 1.0e3 * np.asarray(
                [
                    _float(row, "common_mode_limit_vs_ed_wasserstein_eV")
                    for row in selected
                ]
            )
            if absolute_r.size < 3 or not np.all(np.isfinite(common_wasserstein_meV)):
                raise ValueError(
                    "Incomplete large-splitting/common-mode series for S=%g sign=%+d"
                    % (huang_rhys, sign)
                )
            label = r"$S=%g$, $J%s0$" % (huang_rhys, ">" if sign > 0 else "<")
            axes[1, 0].plot(absolute_r, scba_l1, color=color, linestyle=linestyle, marker=marker, markersize=4, label=label)
            axes[1, 1].plot(absolute_r, common_wasserstein_meV, color=color, linestyle=linestyle, marker=marker, markersize=4)
    for axis in axes[1, :]:
        axis.set_xscale("log", base=2)
        axis.set_xticks((4.0, 8.0, 16.0))
        axis.set_xticklabels(("4", "8", "16"))
        axis.set_xlabel(r"$|r|=2|J|/\omega_0$")
        axis.grid(alpha=0.18, which="both", linewidth=0.5)
    axes[1, 0].set_ylabel(r"SCBA--ED unit-area $L_1$")
    axes[1, 1].set_ylabel("ED--common-mode-limit Wasserstein (meV)")
    axes[1, 0].set_title("SCBA error at large splitting")
    axes[1, 1].set_title(r"Approach to common-mode $S/2$ limit")
    axes[1, 0].legend(frameon=False, fontsize=7.0, ncol=2)
    _panel_label(axes[1, 0], "(c)")
    _panel_label(axes[1, 1], "(d)")

    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markersize=8, label=label)
        for color, label in zip(
            category_colors,
            (r"$L_1<0.25$", r"$0.25\leq L_1<0.75$", r"$0.75\leq L_1<1.25$", r"$L_1\geq1.25$"),
        )
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False)
    fig.suptitle(
        "Operational SCBA regime map (colors are error bands, not universal phase boundaries)",
        y=1.005,
    )
    fig.text(
        0.5,
        0.005,
        r"$J>0$: bright exciton is upper and phonon emission can reach dark; "
        r"$J<0$: bright exciton is lower, so real upward transfer is thermally suppressed at $T=0$. "
        r"Large $|r|$ suppresses difference-mode conversion but leaves common-mode $S/2$ dressing.",
        ha="center",
        va="bottom",
        fontsize=7.6,
    )
    fig.subplots_adjust(wspace=0.24, hspace=0.34, top=0.89, bottom=0.10)
    return _save_figure(fig, output_dir, "regime_map", dpi)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--dpi", type=int, default=240)
    parser.add_argument(
        "--only",
        choices=FIGURE_NAMES,
        action="append",
        help="generate only the named family; repeat for several",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = (MODULE_DIR / results_dir).resolve()
    output_dir = args.figures_dir
    if not output_dir.is_absolute():
        output_dir = (MODULE_DIR / output_dir).resolve()
    if args.dpi < 72:
        raise ValueError("PNG DPI must be at least 72")
    manifest_rows = _read_csv(results_dir / "scan_manifest.csv")
    rows = _read_csv(results_dir / "ed_scba_error_summary.csv")
    convergence_rows = _read_csv(results_dir / "ed_convergence_summary.csv")
    incomplete_manifest = [
        row.get("case_id", "unknown")
        for row in manifest_rows
        if row.get("status") != "complete"
    ]
    if incomplete_manifest:
        preview = ", ".join(incomplete_manifest[:5])
        raise RuntimeError(
            "The production manifest has %d incomplete case(s): %s"
            % (len(incomplete_manifest), preview)
        )
    if any(row.get("manifest_status") != "complete" for row in rows):
        raise RuntimeError("The error summary contains incomplete manifest cases")
    manifest_ids = {row.get("case_id", "") for row in manifest_rows}
    summary_ids = {row.get("case_id", "") for row in rows}
    if manifest_ids != summary_ids or "" in manifest_ids or "" in summary_ids:
        raise RuntimeError(
            "Manifest/error-summary case mismatch: %d manifest IDs versus %d summary IDs"
            % (len(manifest_ids), len(summary_ids))
        )
    selected = set(args.only if args.only is not None else FIGURE_NAMES)
    _configure_style()
    generated: List[Path] = []
    operations = (
        ("direct_analogue_spectra", lambda: plot_coupled_s_spectra(rows, results_dir, output_dir, args.dpi)),
        ("j0_coupling_progression", lambda: plot_j0_progression(rows, results_dir, output_dir, args.dpi)),
        ("error_heatmaps", lambda: plot_error_heatmaps(rows, output_dir, args.dpi)),
        ("signed_j_error_curves", lambda: plot_signed_j_curves(rows, output_dir, args.dpi)),
        ("eta_convergence", lambda: plot_eta_convergence(rows, output_dir, args.dpi)),
        ("self_energy_channels", lambda: plot_self_energy_channels(rows, results_dir, output_dir, args.dpi)),
        ("ed_truncation_convergence", lambda: plot_ed_convergence(convergence_rows, output_dir, args.dpi)),
        ("regime_map", lambda: plot_regime_map(rows, output_dir, args.dpi)),
    )
    for name, operation in operations:
        if name not in selected:
            continue
        paths = operation()
        generated.extend(paths)
        print("Saved %s" % ", ".join(str(path) for path in paths))
    expected = 2 * len(selected)
    if len(generated) != expected or any(not path.is_file() for path in generated):
        raise RuntimeError("Expected %d figure files; generated %d" % (expected, len(generated)))
    print("Generated %d files in %s" % (len(generated), output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
