#!/usr/bin/env python3
"""Create the publication figures for the single-mode frequency scan.

The central figure contains only the dimensionless first Wasserstein distance
``w = W_1 / omega0`` and its bare-molecule ``J=0`` reference.  A systematic
positive-``J`` sweep through the first odd commensurability shows how the
ED--SCBA spectral disagreement develops.  Older figures, including the exact
level map and representative snapshots, are preserved but are not regenerated
or listed as active sources.

All default paths are relative to this file, never to the process working
directory.  The implementation is compatible with Python 3.8.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = MODULE_DIR / "results" / "frequency_scan"
DEFAULT_FIGURE_DIR = MODULE_DIR / "figures" / "frequency_scan"

FIGURE_STEMS = (
    "wasserstein_line_scan",
    "resonance_sweep_spectra",
)
EXACT_LEVEL_STEM = "exact_level_strength_map"
REPRESENTATIVE_STEM = "representative_spectra"

PRIMARY_S = (0.02, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0)
ED_COLOR = "#202020"
SCBA_COLOR = "#0072B2"
ODD_COLOR = "#222222"
EVEN_COLOR = "#A0A0A0"
SELECTED_S = (0.02, 0.05, 0.10)
S_COLORS = {
    0.02: "#009E73",
    0.05: "#0072B2",
    0.10: "#CC79A7",
}
BASELINE_DASH = (0, (6.0, 2.5))
RESONANCE_SWEEP_S = (0.05, 0.10)
RESONANCE_SWEEP_WINDOW = (0.60, 1.50)
RESONANCE_SPECTRAL_WINDOW = (-1.0, 1.5)
RESONANCE_LOWER_FLANK_R = 0.65
RESONANCE_GUIDE_R = 1.00
RESONANCE_CUT_LABELS = ("A", "B", "C", "D")
RESONANCE_CUT_COLORS = ("#009E73", "#D55E00", "#CC79A7", "#E69F00")
RESONANCE_CUT_MARKERS = ("o", "D", "s", "^")

# Filled with exact scan points after the command-15 refinement.  The four
# cases use one S and one J sign: an off-integer minimum, the first odd
# allowed maximum, the even control maximum, and the negative-J counterpart
# of the first odd feature.  Case IDs are exact measured scan coordinates.
REPRESENTATIVE_CASES = (
    ("a", "off-integer background", "primary_S0p05_Jp_r1p45"),
    ("b", "first odd allowed feature", "primary_S0p05_Jp_r0p88"),
    ("c", "even parity-forbidden control", "primary_S0p05_Jp_r1p94"),
    ("d", r"negative-$J$ counterpart", "primary_S0p05_Jm_r0p88"),
)

REPRESENTATIVE_COLUMNS = (
    "display_order", "panel", "case_id", "role", "huang_rhys",
    "j_sign", "j_meV", "r_abs", "omega0_meV", "eta_over_omega0",
    "scba_w_over_omega0", "ed_raw_area", "scba_raw_area",
    "scba_raw_area_relative_difference", "archive_file",
)

RESONANCE_SWEEP_COLUMNS = (
    "display_order", "figure_panel", "cut_label", "role", "selection_rule",
    "case_id", "huang_rhys", "j_sign", "j_meV", "r_abs", "omega0_meV",
    "eta_over_omega0", "scba_w_over_omega0",
    "local_baseline_w_over_omega0", "ed_raw_area", "scba_raw_area",
    "scba_raw_area_relative_difference", "archive_file",
)

LINE_SCAN_SOURCE_COLUMNS = (
    "case_id", "huang_rhys", "j_sign", "j_meV", "r_abs", "omega0_meV",
    "eta_over_omega0", "scba_w_over_omega0",
)

PEAK_MANIFEST_COLUMNS = (
    "huang_rhys", "j_sign", "integer_guide", "feature_label",
    "extremum_type", "r_peak", "w_peak", "peak_case_id",
    "r_reference_min", "w_reference_min", "reference_case_id",
    "reference_label", "search_window", "reference_window",
)

SOURCE_MANIFEST_COLUMNS = (
    "figure_stem", "panels", "output_png", "output_pdf", "source_file",
    "source_sha256", "source_size_bytes", "selection", "plotted_quantities",
    "transformation", "notes",
)


def _configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 8.3,
        "ytick.labelsize": 8.3,
        "legend.fontsize": 8.0,
        "lines.linewidth": 1.35,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError("Required CSV is missing: %s" % path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header: %s" % path)
        return [dict(row) for row in reader]


def _write_csv_atomic(path: Path, rows: Iterable[Mapping[str, Any]],
                      columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=str(path.parent),
            prefix=path.name + ".", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in columns})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(path))
    except Exception:
        if temporary is not None:
            try:
                Path(temporary).unlink()
            except FileNotFoundError:
                pass
        raise


def _float(row: Mapping[str, str], key: str) -> float:
    text = str(row.get(key, "")).strip()
    if not text:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _bool(row: Mapping[str, str], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in ("true", "1", "yes")


def _close(left: float, right: float, tolerance: float = 1.0e-10) -> bool:
    return bool(np.isfinite(left) and abs(left - right) <= tolerance)


def _panel_label(axis: plt.Axes, label: str, color: str = "black") -> None:
    axis.text(
        0.015, 0.975, "(%s)" % label, transform=axis.transAxes,
        ha="left", va="top", fontsize=10.0, fontweight="bold", color=color,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.68, pad=0.8),
        zorder=20,
    )


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(MODULE_DIR))
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str,
                 dpi: int) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = output_dir / (stem + ".pdf")
    png = output_dir / (stem + ".png")
    fig.savefig(str(pdf), bbox_inches="tight", pad_inches=0.045)
    fig.savefig(str(png), dpi=dpi, bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)
    return png.resolve(), pdf.resolve()


def _validate_inputs(results_dir: Path) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, str]]]:
    summary_path = results_dir / "summary.csv"
    manifest_path = results_dir / "manifest.csv"
    validation_path = results_dir.parent.parent / "validation" / "frequency_scan" / "validation.json"
    summary = _read_csv(summary_path)
    manifest = _read_csv(manifest_path)
    if not manifest or any(row.get("status") != "complete" for row in manifest):
        raise RuntimeError("Frequency-scan manifest is not wholly complete")
    summary_by_id = {row.get("case_id", ""): row for row in summary}
    manifest_ids = {row.get("case_id", "") for row in manifest}
    if "" in summary_by_id or len(summary_by_id) != len(summary):
        raise ValueError("Summary contains empty or duplicate case IDs")
    if manifest_ids != set(summary_by_id):
        raise RuntimeError("Manifest and summary case-ID sets disagree")
    if not validation_path.is_file():
        raise FileNotFoundError("Numerical validation JSON is missing: %s" % validation_path)
    with validation_path.open("r", encoding="utf-8") as handle:
        validation = json.load(handle)
    if not bool(validation.get("pass")):
        raise RuntimeError("Frequency-scan numerical validation does not pass")
    return summary, summary_by_id


def _primary_grid(summary: Sequence[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[float, int, float], Dict[str, str]]]:
    rows = [row for row in summary if row.get("family") == "primary_fixed_S"]
    s_values = np.asarray(sorted({_float(row, "huang_rhys") for row in rows}), dtype=float)
    r_values = np.asarray(sorted({_float(row, "r_abs") for row in rows}), dtype=float)
    if s_values.size != len(PRIMARY_S) or not np.allclose(s_values, PRIMARY_S, rtol=0.0, atol=1.0e-12):
        raise ValueError("Primary S grid differs from the declared seven values")
    lookup: Dict[Tuple[float, int, float], Dict[str, str]] = {}
    for row in rows:
        key = (round(_float(row, "huang_rhys"), 12), int(round(_float(row, "j_sign"))),
               round(_float(row, "r_abs"), 12))
        if key in lookup:
            raise ValueError("Duplicate primary scan cell: %r" % (key,))
        lookup[key] = row
    expected = len(s_values) * len(r_values) * 2
    if len(lookup) != expected:
        raise ValueError("Primary scan is incomplete: %d cells, expected %d" % (len(lookup), expected))
    return s_values, r_values, lookup


def _draw_integer_guides(axis: plt.Axes) -> None:
    for value in (1.0, 3.0, 5.0):
        axis.axvline(value, color=ODD_COLOR, linestyle="--", linewidth=0.75,
                     alpha=0.75, zorder=5)
    for value in (2.0, 4.0):
        axis.axvline(value, color=EVEN_COLOR, linestyle=":", linewidth=0.8,
                     alpha=0.90, zorder=5)


def extract_peak_manifest(summary: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Record measured guide-window maxima and nearby background minima.

    A window maximum is called a local maximum only when its immediate
    measured neighbours are lower.  This distinction prevents a monotonic
    background from being promoted to a commensurability cusp merely because
    an integer-centred search window was requested.
    """

    primary = [row for row in summary if row.get("family") == "primary_fixed_S"]
    groups: Dict[Tuple[float, int], List[Dict[str, str]]] = {}
    for s_value in SELECTED_S:
        for sign in (-1, +1):
            values = [
                row for row in primary
                if _close(_float(row, "huang_rhys"), s_value)
                and int(round(_float(row, "j_sign"))) == sign
            ]
            values.sort(key=lambda row: _float(row, "r_abs"))
            if not values:
                raise ValueError("Missing peak-manifest group S=%g sign=%d" % (s_value, sign))
            groups[(s_value, sign)] = values

    output: List[Dict[str, Any]] = []
    for (s_value, sign), values in sorted(groups.items()):
        positions = [_float(row, "r_abs") for row in values]
        for guide in range(1, 6):
            core_window = [
                row for row in values
                if abs(_float(row, "r_abs") - guide) <= 0.15 + 1.0e-12
            ]
            reference = [
                row for row in values
                if 0.15 + 1.0e-12 < abs(_float(row, "r_abs") - guide)
                <= 0.50 + 1.0e-12
            ]
            if not core_window or not reference:
                raise ValueError("Incomplete extrema window S=%g sign=%d n=%d" %
                                 (s_value, sign, guide))
            local_maxima = [
                values[index] for index in range(1, len(values) - 1)
                if abs(positions[index] - guide) < 0.50 - 1.0e-12
                and _float(values[index], "scba_w_over_omega0")
                > _float(values[index - 1], "scba_w_over_omega0")
                and _float(values[index], "scba_w_over_omega0")
                > _float(values[index + 1], "scba_w_over_omega0")
            ]
            peak = (max(local_maxima, key=lambda row: _float(row, "scba_w_over_omega0"))
                    if local_maxima else
                    max(core_window, key=lambda row: _float(row, "scba_w_over_omega0")))
            background = min(reference, key=lambda row: _float(row, "scba_w_over_omega0"))
            is_local = bool(local_maxima)
            output.append({
                "huang_rhys": s_value,
                "j_sign": sign,
                "integer_guide": guide,
                "feature_label": ("odd allowed guide" if guide % 2
                                  else "even parity-forbidden control"),
                "extremum_type": ("resolved local maximum" if is_local
                                  else "guide-window maximum; no resolved local cusp"),
                "r_peak": _float(peak, "r_abs"),
                "w_peak": _float(peak, "scba_w_over_omega0"),
                "peak_case_id": peak.get("case_id", ""),
                "r_reference_min": _float(background, "r_abs"),
                "w_reference_min": _float(background, "scba_w_over_omega0"),
                "reference_case_id": background.get("case_id", ""),
                "reference_label": "noninteger background",
                "search_window": "|r-n| < 0.50; strict local maxima preferred",
                "reference_window": "0.15 < |r-n| <= 0.50",
            })
    return output


def plot_wasserstein_line_scan(
    summary: Sequence[Dict[str, str]], output_dir: Path, dpi: int,
) -> Tuple[Path, Path]:
    s_values, r_values, lookup = _primary_grid(summary)
    del s_values
    baselines = sorted(
        (row for row in summary if row.get("family") == "primary_J0_baseline"),
        key=lambda row: _float(row, "huang_rhys"),
    )
    if len(baselines) != len(PRIMARY_S):
        raise ValueError("The seven fixed-S local baselines are required")
    baseline_by_s = {
        round(_float(row, "huang_rhys"), 12): _float(row, "scba_w_over_omega0")
        for row in baselines
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), sharex=True, sharey=True)
    signs = (+1, -1)
    maximum = 0.0
    for panel, (axis, sign) in enumerate(zip(axes, signs)):
        _draw_integer_guides(axis)
        for s_value in SELECTED_S:
            w_values = np.asarray([
                _float(lookup[(round(s_value, 12), sign, round(float(r_value), 12))],
                       "scba_w_over_omega0")
                for r_value in r_values
            ])
            maximum = max(maximum, float(np.max(w_values)))
            axis.plot(
                r_values, w_values, color=S_COLORS[s_value],
                linewidth=1.35, marker=".", markersize=1.45,
                label=r"$S=%g$" % s_value, zorder=4,
            )
            baseline = baseline_by_s[round(s_value, 12)]
            maximum = max(maximum, baseline)
            axis.axhline(
                baseline, color=S_COLORS[s_value], linestyle=BASELINE_DASH,
                linewidth=1.05, alpha=0.82, zorder=3,
            )
        axis.set_xlim(0.25, 5.5)
        axis.set_xticks((0.5, 1, 2, 3, 4, 5, 5.5))
        axis.set_xlabel(r"$r=2|J|/\omega_0$")
        axis.set_title(
            (r"$J=+25$ meV: upper bright state" if sign > 0
             else r"$J=-25$ meV: lower bright state"),
            pad=6.0,
        )
        axis.grid(alpha=0.16, linewidth=0.45)
        _panel_label(axis, chr(ord("a") + panel))
    axes[0].set_ylabel(r"$w=W_1(A_{\rm ED},A_{\rm SCBA})/\omega_0$")
    axes[0].set_ylim(0.0, 1.08 * maximum)

    inset = axes[1].inset_axes([0.53, 0.49, 0.44, 0.43])
    baseline_s = np.asarray([_float(row, "huang_rhys") for row in baselines])
    baseline_w = np.asarray([_float(row, "scba_w_over_omega0") for row in baselines])
    inset.plot(baseline_s, baseline_w, color="#555555", marker="o",
               markersize=2.8, linewidth=1.0)
    for s_value in SELECTED_S:
        index = int(np.argmin(np.abs(baseline_s - s_value)))
        inset.scatter([baseline_s[index]], [baseline_w[index]],
                      color=S_COLORS[s_value], s=18.0, zorder=5)
    inset.set_xscale("log")
    inset.set_yscale("log")
    inset.set_xlabel(r"$S$", fontsize=7.2, labelpad=0.5)
    inset.set_ylabel(r"$w_0(S)$", fontsize=7.2, labelpad=0.5)
    inset.set_title(r"local $J=0$ baseline", fontsize=7.5, pad=2.0)
    inset.tick_params(labelsize=6.6, pad=1.0)
    inset.grid(alpha=0.15, linewidth=0.35)

    curve_handles = [
        Line2D([0], [0], color=S_COLORS[s_value], linewidth=1.5,
               label=r"$S=%g$" % s_value)
        for s_value in SELECTED_S
    ]
    guide_handles = [
        Line2D([0], [0], color="#555555", linestyle=BASELINE_DASH,
               linewidth=1.15, label=r"same-color dashed: $J=0$ baseline"),
        Line2D([0], [0], color=ODD_COLOR, linestyle="--", linewidth=1.0,
               label="odd allowed guide"),
        Line2D([0], [0], color=EVEN_COLOR, linestyle=":", linewidth=1.1,
               label="even parity-forbidden control"),
    ]
    fig.legend(handles=curve_handles + guide_handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.015), ncol=6, frameon=False,
               columnspacing=1.05, handlelength=2.5)
    fig.subplots_adjust(left=0.083, right=0.99, bottom=0.20, top=0.91, wspace=0.08)
    return _save_figure(fig, output_dir, FIGURE_STEMS[0], dpi)


def _parity_branches(energy: np.ndarray, parity: np.ndarray,
                     parity_value: int) -> np.ndarray:
    counts = [int(np.sum(parity[index] == parity_value)) for index in range(energy.shape[0])]
    branches = np.full((energy.shape[0], max(counts)), np.nan, dtype=float)
    for index in range(energy.shape[0]):
        values = np.sort(energy[index, parity[index] == parity_value])
        branches[index, :values.size] = values
    return branches


def _finite(text: str) -> float:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return float("nan")
    return value


def plot_exact_level_strength_map(
    results_dir: Path, output_dir: Path, dpi: int,
) -> Tuple[Path, Path]:
    archive_path = results_dir / "exact_level_map.npz"
    crossings_path = results_dir / "exact_level_map_crossings.csv"
    if not archive_path.is_file():
        raise FileNotFoundError("Exact level archive is missing: %s" % archive_path)
    crossing_rows = _read_csv(crossings_path)
    with np.load(str(archive_path), allow_pickle=False) as archive:
        required = ("huang_rhys", "r_abs", "detuning_from_bare_bright_over_omega0",
                    "bright0_weight", "parity", "metadata_json")
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise KeyError("Exact level archive lacks %s" % missing)
        s_values = np.asarray(archive["huang_rhys"], dtype=float)
        r_values = np.asarray(archive["r_abs"], dtype=float)
        energy = np.asarray(archive["detuning_from_bare_bright_over_omega0"], dtype=float)
        bright_weight = np.asarray(archive["bright0_weight"], dtype=float)
        parity = np.asarray(archive["parity"], dtype=int)
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    if tuple(np.round(s_values, 12)) != (0.1, 0.5, 1.0):
        raise ValueError("Exact level archive does not contain S=0.1,0.5,1")
    if energy.shape != bright_weight.shape or energy.shape != parity.shape:
        raise ValueError("Exact level arrays have inconsistent shapes")
    if int(metadata.get("schema_version", -1)) != 1:
        raise ValueError("Unexpected exact-level schema")

    fig = plt.figure(figsize=(12.6, 4.55))
    grid = fig.add_gridspec(1, 4, width_ratios=(1.0, 1.0, 1.0, 0.045),
                            wspace=0.10)
    axes_list: List[plt.Axes] = []
    for index in range(3):
        axes_list.append(fig.add_subplot(
            grid[0, index],
            sharex=axes_list[0] if axes_list else None,
            sharey=axes_list[0] if axes_list else None,
        ))
    axes = np.asarray(axes_list)
    colorbar_axis = fig.add_subplot(grid[0, 3])
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("inferno")
    scatter = None
    for s_index, (axis, s_value) in enumerate(zip(axes, s_values)):
        y = energy[s_index]
        p = parity[s_index]
        for parity_value, color, linestyle in (
            (+1, "#7797B7", "-"), (-1, "#B7B7B7", ":"),
        ):
            branches = _parity_branches(y, p, parity_value)
            for rank in range(branches.shape[1]):
                branch = branches[:, rank]
                if np.any(np.isfinite(branch) & (branch >= -1.35) & (branch <= 1.35)):
                    axis.plot(r_values, branch, color=color, linestyle=linestyle,
                              linewidth=0.58, alpha=0.78, zorder=1)
        weight = bright_weight[s_index]
        r_mesh = np.repeat(r_values[:, None], weight.shape[1], axis=1)
        selected = weight > 2.0e-6
        scatter = axis.scatter(
            r_mesh[selected], y[selected], c=weight[selected], cmap=cmap, norm=norm,
            s=4.0 + 62.0 * np.sqrt(weight[selected]), edgecolors="none",
            alpha=0.92, rasterized=True, zorder=4,
        )
        _draw_integer_guides(axis)
        axis.axhline(0.0, color="#555555", linewidth=0.55, alpha=0.65, zorder=0)
        relevant_crossings = [
            row for row in crossing_rows
            if _close(_float(row, "huang_rhys"), float(s_value))
        ]
        for row in relevant_crossings:
            # A boundary-pinned or identity-ambiguous minimum is not a
            # defensible crossing center and is deliberately not marked.
            if _bool(row, "tracked_gap_ambiguous"):
                continue
            if row.get("crossing_type") == "odd_allowed":
                x_value = _finite(row.get("tracked_min_r", ""))
                marker = "v"; color = "#111111"
            else:
                x_value = _finite(row.get("exact_control_crossing_r", ""))
                marker = "x"; color = "#777777"
            center = _finite(row.get("tracked_min_cluster_centroid_over_omega0", ""))
            if np.isfinite(x_value) and np.isfinite(center):
                y_value = center - 0.5 * x_value
                if -1.35 <= y_value <= 1.35:
                    axis.scatter([x_value], [y_value], marker=marker, s=31.0,
                                 color=color, linewidths=0.9, zorder=8)
        forbidden = [
            abs(_finite(row.get("forbidden_block_coupling_over_omega0", "")))
            for row in relevant_crossings
            if row.get("crossing_type") == "even_forbidden_control"
            and np.isfinite(_finite(row.get("forbidden_block_coupling_over_omega0", "")))
        ]
        maximum_forbidden = max(forbidden) if forbidden else float("nan")
        axis.text(
            0.975, 0.965,
            r"$S=%g$" "\n" r"$\max|H_{+-}|/\omega_0=%.1e$" % (s_value, maximum_forbidden),
            transform=axis.transAxes, ha="right", va="top", fontsize=8.1,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=1.4),
            zorder=10,
        )
        axis.set_xlim(0.25, 5.5)
        axis.set_ylim(-1.35, 1.35)
        axis.set_xticks((0.5, 1, 2, 3, 4, 5, 5.5))
        if s_index > 0:
            axis.tick_params(axis="y", labelleft=False)
        axis.set_xlabel(r"$r=2|J|/\omega_0$")
        axis.grid(alpha=0.10, linewidth=0.45)
        _panel_label(axis, chr(ord("a") + s_index))
    axes[0].set_ylabel(r"Exact level detuning, $(E_\nu-E_B)/\omega_0$")
    if scatter is None:
        raise RuntimeError("Exact level plot has no bright-weight points")
    colorbar = fig.colorbar(scatter, cax=colorbar_axis)
    colorbar.set_label(r"Bare-bright weight $|\langle B,0|\nu\rangle|^2$")
    legend_handles = (
        Line2D([0], [0], color="#7797B7", linestyle="-", linewidth=1.0,
               label=r"$\Pi=+1$ sector"),
        Line2D([0], [0], color="#B7B7B7", linestyle=":", linewidth=1.1,
               label=r"$\Pi=-1$ sector"),
        Line2D([0], [0], color=ODD_COLOR, linestyle="--", linewidth=1.0,
               label="odd allowed guide"),
        Line2D([0], [0], color=EVEN_COLOR, linestyle=":", linewidth=1.1,
               label="even parity-forbidden control"),
        Line2D([0], [0], marker="v", color="#111111", linestyle="none",
               markersize=5.0, label="tracked odd avoided crossing"),
        Line2D([0], [0], marker="x", color="#777777", linestyle="none",
               markersize=5.0, label="tracked exact even crossing"),
    )
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.49, -0.005),
               ncol=3, frameon=False, columnspacing=1.6)
    fig.suptitle(
        r"Parity-resolved difference-mode levels: oscillator strength transfers "
        r"only within $\Pi=+1$" "\n"
        r"$J=+25$ meV, $\omega_0=50/r$ meV; common mode omitted",
        y=1.03,
    )
    fig.subplots_adjust(bottom=0.235, top=0.88, left=0.072, right=0.955)
    return _save_figure(fig, output_dir, EXACT_LEVEL_STEM, dpi)


def _archive_path(row: Mapping[str, str], results_dir: Path) -> Path:
    text = str(row.get("archive_file", "")).strip()
    candidates = []
    if text:
        candidate = Path(text)
        candidates.append(candidate if candidate.is_absolute() else MODULE_DIR / candidate)
    candidates.append(results_dir / "spectra" / (str(row["case_id"]) + ".npz"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Representative archive is missing for %s" % row.get("case_id"))


def _load_spectral_archive(path: Path, expected_case_id: str) -> Dict[str, Any]:
    required = (
        "energy_eV", "analysis_mask", "ed_green_bright_per_eV",
        "ba_green_bright_per_eV", "scba_green_bright_per_eV", "metadata_json",
    )
    with np.load(str(path), allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise KeyError("Archive %s lacks %s" % (path, missing))
        result: Dict[str, Any] = {
            "energy_eV": np.asarray(archive["energy_eV"], dtype=float).copy(),
            "analysis_mask": np.asarray(archive["analysis_mask"], dtype=bool).copy(),
            "ed": np.asarray(archive["ed_green_bright_per_eV"], dtype=complex).copy(),
            "ba": np.asarray(archive["ba_green_bright_per_eV"], dtype=complex).copy(),
            "scba": np.asarray(archive["scba_green_bright_per_eV"], dtype=complex).copy(),
            "metadata": json.loads(str(np.asarray(archive["metadata_json"]).item())),
        }
    energy = result["energy_eV"]
    if result["metadata"].get("case_id") != expected_case_id:
        raise ValueError("Archive case-ID mismatch: %s" % path)
    if result["analysis_mask"].shape != energy.shape:
        raise ValueError("Archive mask shape mismatch: %s" % path)
    for key in ("ed", "ba", "scba"):
        if result[key].shape != energy.shape or not np.all(np.isfinite(result[key])):
            raise ValueError("Invalid %s array in %s" % (key, path))
    return result


def _dimensionless_unit_area(
    green: np.ndarray, energy: np.ndarray, mask: np.ndarray, omega0_eV: float,
) -> Tuple[np.ndarray, float]:
    spectrum = -np.imag(green)
    if float(np.min(spectrum[mask])) < -1.0e-10:
        raise ValueError("A representative spectrum is negative")
    raw_area = float(np.trapz(spectrum[mask], energy[mask]))
    if not np.isfinite(raw_area) or raw_area <= 0.0:
        raise ValueError("Cannot normalize a nonpositive representative spectrum")
    return omega0_eV * spectrum / raw_area, raw_area


def _representative_rows(
    summary_by_id: Mapping[str, Dict[str, str]], results_dir: Path,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for order, (panel, role, case_id) in enumerate(REPRESENTATIVE_CASES, start=1):
        if case_id not in summary_by_id:
            raise KeyError("Representative case is missing from summary: %s" % case_id)
        source = summary_by_id[case_id]
        archive = _archive_path(source, results_dir)
        if abs(_float(source, "scba_raw_area_relative_difference")) > 3.0e-6:
            raise ValueError("Representative raw-area check failed: %s" % case_id)
        row: Dict[str, Any] = {key: source.get(key, "") for key in REPRESENTATIVE_COLUMNS}
        row.update({
            "display_order": order, "panel": panel, "role": role,
            "archive_file": _portable_path(archive),
        })
        rows.append(row)
    return rows


def plot_selected_frequency_spectra(
    summary_by_id: Mapping[str, Dict[str, str]], results_dir: Path,
    output_dir: Path, dpi: int,
) -> Tuple[Tuple[Path, Path], List[Dict[str, Any]], List[Path]]:
    representative = _representative_rows(summary_by_id, results_dir)
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.25), sharex=True, sharey=True)
    source_archives: List[Path] = []
    maximum_raw_mismatch = 0.0
    maximum_display = 0.0
    legend_handles: Optional[Tuple[Line2D, Line2D]] = None
    for axis, table_row in zip(np.ravel(axes), representative):
        case_id = str(table_row["case_id"])
        summary = summary_by_id[case_id]
        archive_path = _archive_path(summary, results_dir)
        source_archives.append(archive_path)
        data = _load_spectral_archive(archive_path, case_id)
        model = data["metadata"]["model"]
        omega0_eV = float(model["omega0_meV"]) / 1000.0
        bright_energy_eV = (
            float(model["energy_00_eV"])
            + float(model["huang_rhys"]) * omega0_eV
            + float(model["j_meV"]) / 1000.0
        )
        energy = data["energy_eV"]
        mask = data["analysis_mask"]
        detuning = (energy - bright_energy_eV) / omega0_eV
        display = mask & (detuning >= -4.0) & (detuning <= 4.0)
        ed, area_ed = _dimensionless_unit_area(data["ed"], energy, mask, omega0_eV)
        scba, area_scba = _dimensionless_unit_area(data["scba"], energy, mask, omega0_eV)
        maximum_raw_mismatch = max(
            maximum_raw_mismatch,
            abs(area_scba / area_ed - 1.0),
        )
        maximum_display = max(
            maximum_display, float(np.max(ed[display])), float(np.max(scba[display]))
        )
        line_ed, = axis.plot(detuning[display], ed[display], color=ED_COLOR,
                             linewidth=1.45, label="ED", zorder=5)
        line_scba, = axis.plot(detuning[display], scba[display], color=SCBA_COLOR,
                               linewidth=1.25, label="SCBA", alpha=0.95)
        legend_handles = (line_ed, line_scba)
        axis.axvline(0.0, color="#888888", linestyle=":", linewidth=0.65)
        axis.grid(alpha=0.13, linewidth=0.45)
        axis.set_xlim(-4.0, 4.0)
        sign_text = "+" if int(round(_float(summary, "j_sign"))) > 0 else "-"
        axis.set_title(str(table_row["role"]), fontsize=8.6, pad=5.0)
        axis.text(
            0.975, 0.93,
            r"$S=%g$, $\mathrm{sgn}(J)=%s$" "\n" r"$r=%.3f$" "\n" r"$w=%.5f$"
            % (_float(summary, "huang_rhys"), sign_text,
               _float(summary, "r_abs"), _float(summary, "scba_w_over_omega0")),
            transform=axis.transAxes, ha="right", va="top", fontsize=7.7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=1.2),
            zorder=10,
        )
        _panel_label(axis, str(table_row["panel"]))
    for axis in np.ravel(axes):
        axis.set_xlabel(r"$(E-E_B)/\omega_0$")
    axes[0].set_ylim(0.0, 1.04 * maximum_display)
    axes[0].set_ylabel(r"Unit-area $\omega_0 A_B(E)$")
    if legend_handles is not None:
        fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.955),
                   ncol=2, frameon=False)
    fig.suptitle(
        r"Resolved bright absorption: $S=0.05$, $|J|=25$ meV, "
        r"$T=0$, $\eta/\omega_0=0.01$",
        y=1.06, fontsize=10.2,
    )
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.16, top=0.78,
                        wspace=0.08)
    paths = _save_figure(fig, output_dir, REPRESENTATIVE_STEM, dpi)
    return paths, representative, source_archives


def _measured_row_at(
    rows: Sequence[Dict[str, str]], r_target: float,
) -> Dict[str, str]:
    matches = [row for row in rows if _close(_float(row, "r_abs"), r_target)]
    if len(matches) != 1:
        raise ValueError(
            "Expected one measured row at r=%.6g, found %d" %
            (r_target, len(matches))
        )
    return matches[0]


def _resonance_sweep_rows(
    summary: Sequence[Dict[str, str]],
    summary_by_id: Mapping[str, Dict[str, str]],
    peak_rows: Sequence[Mapping[str, Any]],
    results_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Path]]:
    """Select the four ordered, measured cuts used for each first-resonance sweep."""

    baselines = {
        round(_float(row, "huang_rhys"), 12): row
        for row in summary if row.get("family") == "primary_J0_baseline"
    }
    output: List[Dict[str, Any]] = []
    archives: List[Path] = []
    missing: List[Tuple[str, Path]] = []
    display_order = 0
    for s_index, s_value in enumerate(RESONANCE_SWEEP_S):
        primary = sorted(
            (
                row for row in summary
                if row.get("family") == "primary_fixed_S"
                and _close(_float(row, "huang_rhys"), s_value)
                and int(round(_float(row, "j_sign"))) == +1
            ),
            key=lambda row: _float(row, "r_abs"),
        )
        if not primary:
            raise ValueError("Missing positive-J sweep rows for S=%g" % s_value)
        baseline = baselines.get(round(s_value, 12))
        if baseline is None:
            raise ValueError("Missing J=0 local baseline for S=%g" % s_value)
        guide_record = [
            row for row in peak_rows
            if _close(_float(row, "huang_rhys"), s_value)
            and int(round(_float(row, "j_sign"))) == +1
            and int(round(_float(row, "integer_guide"))) == 1
        ]
        if len(guide_record) != 1:
            raise ValueError("Expected one first-guide peak record for S=%g" % s_value)
        peak_record = guide_record[0]
        if str(peak_record.get("extremum_type", "")) != "resolved local maximum":
            raise ValueError("The positive-J first feature is not a resolved maximum at S=%g" % s_value)
        peak_id = str(peak_record.get("peak_case_id", ""))
        reference_id = str(peak_record.get("reference_case_id", ""))
        if peak_id not in summary_by_id or reference_id not in summary_by_id:
            raise KeyError("Peak-manifest case IDs are absent from the scan summary")

        selected = (
            (
                "lower flank", "shared measured lower flank r=0.65",
                _measured_row_at(primary, RESONANCE_LOWER_FLANK_R),
            ),
            (
                "measured maximum",
                "resolved first local maximum from the measured peak manifest",
                summary_by_id[peak_id],
            ),
            (
                "bare guide",
                "measured bare first-odd commensurability guide r=1.00",
                _measured_row_at(primary, RESONANCE_GUIDE_R),
            ),
            (
                "upper off-integer reference minimum",
                "measured minimum in 0.15<|r-1|<=0.50 from the peak manifest",
                summary_by_id[reference_id],
            ),
        )
        selected_r = [_float(item[2], "r_abs") for item in selected]
        if not all(selected_r[index] < selected_r[index + 1]
                   for index in range(len(selected_r) - 1)):
            raise ValueError(
                "First-resonance cuts are not ordered below/peak/guide/above "
                "for S=%g: %s" % (s_value, selected_r)
            )

        for cut_index, (role, rule, source) in enumerate(selected):
            display_order += 1
            case_id = str(source.get("case_id", ""))
            if abs(_float(source, "scba_raw_area_relative_difference")) > 3.0e-6:
                raise ValueError("Sweep raw-area check failed: %s" % case_id)
            try:
                archive = _archive_path(source, results_dir)
            except FileNotFoundError:
                expected = (results_dir / "spectra" / (case_id + ".npz")).resolve()
                missing.append((case_id, expected))
                archive = expected
            row: Dict[str, Any] = {
                key: source.get(key, "") for key in RESONANCE_SWEEP_COLUMNS
            }
            row.update({
                "display_order": display_order,
                "figure_panel": chr(ord("c") + s_index),
                "cut_label": RESONANCE_CUT_LABELS[cut_index],
                "role": role,
                "selection_rule": rule,
                "local_baseline_w_over_omega0": _float(
                    baseline, "scba_w_over_omega0"
                ),
                "archive_file": _portable_path(archive),
            })
            output.append(row)
            archives.append(archive)

    if missing:
        detail = "\n".join(
            "  %s -> %s" % (case_id, path) for case_id, path in missing
        )
        raise FileNotFoundError(
            "The resonance sweep requires exact spectral archives for these "
            "measured cases:\n%s" % detail
        )
    return output, archives


def plot_resonance_sweep_spectra(
    summary: Sequence[Dict[str, str]],
    summary_by_id: Mapping[str, Dict[str, str]],
    peak_rows: Sequence[Mapping[str, Any]],
    results_dir: Path,
    output_dir: Path,
    dpi: int,
) -> Tuple[Tuple[Path, Path], List[Dict[str, Any]], List[Path]]:
    """Plot measured error traces and four stacked ED--SCBA cuts per S."""

    sweep_rows, source_archives = _resonance_sweep_rows(
        summary, summary_by_id, peak_rows, results_dir
    )
    fig = plt.figure(figsize=(11.8, 7.4))
    grid = fig.add_gridspec(
        2, 2, height_ratios=(0.92, 2.25), hspace=0.31, wspace=0.31
    )
    scan_axes = [fig.add_subplot(grid[0, index]) for index in range(2)]
    spectrum_axes = [fig.add_subplot(grid[1, index]) for index in range(2)]

    for s_index, (axis, s_value) in enumerate(zip(scan_axes, RESONANCE_SWEEP_S)):
        trace = sorted(
            (
                row for row in summary
                if row.get("family") == "primary_fixed_S"
                and _close(_float(row, "huang_rhys"), s_value)
                and int(round(_float(row, "j_sign"))) == +1
                and RESONANCE_SWEEP_WINDOW[0] - 1.0e-12
                <= _float(row, "r_abs")
                <= RESONANCE_SWEEP_WINDOW[1] + 1.0e-12
            ),
            key=lambda row: _float(row, "r_abs"),
        )
        cuts = [
            row for row in sweep_rows
            if _close(_float(row, "huang_rhys"), s_value)
        ]
        if not trace or len(cuts) != 4:
            raise ValueError("Incomplete first-resonance plot selection for S=%g" % s_value)
        r_values = np.asarray([_float(row, "r_abs") for row in trace])
        w_values = np.asarray([_float(row, "scba_w_over_omega0") for row in trace])
        baseline = _float(cuts[0], "local_baseline_w_over_omega0")
        axis.plot(
            r_values, w_values, color=S_COLORS[s_value], linewidth=1.45,
            marker=".", markersize=2.3, zorder=4,
        )
        axis.axhline(
            baseline, color=S_COLORS[s_value], linestyle=BASELINE_DASH,
            linewidth=1.05, alpha=0.85, zorder=2,
        )
        axis.axvline(
            RESONANCE_GUIDE_R, color=ODD_COLOR, linestyle=":",
            linewidth=0.95, alpha=0.85, zorder=2,
        )
        for cut_index, row in enumerate(cuts):
            r_cut = _float(row, "r_abs")
            w_cut = _float(row, "scba_w_over_omega0")
            axis.scatter(
                [r_cut], [w_cut], marker=RESONANCE_CUT_MARKERS[cut_index],
                s=38.0, facecolor=RESONANCE_CUT_COLORS[cut_index],
                edgecolor="white", linewidth=0.65, zorder=8,
            )
            axis.annotate(
                str(row["cut_label"]), (r_cut, w_cut), xytext=(0, 6),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7.8, fontweight="bold",
                color=RESONANCE_CUT_COLORS[cut_index], zorder=9,
            )
        lower = min(float(np.min(w_values)), baseline)
        upper = max(float(np.max(w_values)), baseline)
        padding = max(0.10 * (upper - lower), 3.0e-4)
        axis.set_xlim(*RESONANCE_SWEEP_WINDOW)
        axis.set_ylim(max(0.0, lower - padding), upper + 1.55 * padding)
        axis.set_xlabel(r"$r=2J/\omega_0$")
        axis.set_ylabel(r"$w=W_1/\omega_0$")
        axis.set_title(r"$J=+25$ meV, $S=%g$" % s_value, pad=5.0)
        axis.grid(alpha=0.14, linewidth=0.42)
        _panel_label(axis, chr(ord("a") + s_index))

    scan_legend = (
        Line2D([0], [0], color="#555555", marker=".", markersize=3.0,
               linewidth=1.4, label=r"$w(r)$"),
        Line2D([0], [0], color="#555555", linestyle=BASELINE_DASH,
               linewidth=1.1, label="baseline"),
        Line2D([0], [0], color=ODD_COLOR, linestyle=":", linewidth=1.0,
               label=r"$r=1$"),
    )
    fig.legend(
        handles=scan_legend, loc="upper center", bbox_to_anchor=(0.5, 0.925),
        ncol=3, frameon=False, columnspacing=1.8, handlelength=2.5,
    )

    spectral_legend: Optional[Tuple[Line2D, Line2D]] = None
    for s_index, (axis, s_value) in enumerate(zip(spectrum_axes, RESONANCE_SWEEP_S)):
        group = [
            (row, archive) for row, archive in zip(sweep_rows, source_archives)
            if _close(_float(row, "huang_rhys"), s_value)
        ]
        if len(group) != 4:
            raise ValueError("Expected four spectral cuts for S=%g" % s_value)
        prepared: List[Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = []
        maximum = 0.0
        for row, archive_path in group:
            case_id = str(row["case_id"])
            data = _load_spectral_archive(archive_path, case_id)
            model = data["metadata"]["model"]
            omega0_eV = float(model["omega0_meV"]) / 1000.0
            bright_energy_eV = (
                float(model["energy_00_eV"])
                + float(model["huang_rhys"]) * omega0_eV
                + float(model["j_meV"]) / 1000.0
            )
            energy = data["energy_eV"]
            mask = data["analysis_mask"]
            detuning = (energy - bright_energy_eV) / omega0_eV
            display = (
                mask
                & (detuning >= RESONANCE_SPECTRAL_WINDOW[0])
                & (detuning <= RESONANCE_SPECTRAL_WINDOW[1])
            )
            if not np.any(display):
                raise ValueError("Empty display window for %s" % case_id)
            ed, area_ed = _dimensionless_unit_area(
                data["ed"], energy, mask, omega0_eV
            )
            scba, area_scba = _dimensionless_unit_area(
                data["scba"], energy, mask, omega0_eV
            )
            if abs(area_scba / area_ed - 1.0) > 3.0e-6:
                raise ValueError("Archived sweep raw-area check failed: %s" % case_id)
            maximum = max(
                maximum, float(np.max(ed[display])), float(np.max(scba[display]))
            )
            prepared.append((row, detuning[display], ed[display], scba[display]))
        spacing = 1.22 * maximum
        offsets = np.arange(4, dtype=float) * spacing
        tick_labels: List[str] = []
        for cut_index, ((row, detuning, ed, scba), offset) in enumerate(
            zip(prepared, offsets)
        ):
            line_ed, = axis.plot(
                detuning, offset + ed, color=ED_COLOR, linewidth=1.30,
                zorder=5, label="ED",
            )
            line_scba, = axis.plot(
                detuning, offset + scba, color=SCBA_COLOR, linewidth=1.15,
                alpha=0.96, zorder=4, label="SCBA",
            )
            spectral_legend = (line_ed, line_scba)
            axis.axhline(offset, color="#BBBBBB", linewidth=0.45, zorder=1)
            tick_labels.append(
                "%s  r=%.3f, w=%.4f" % (
                    row["cut_label"], _float(row, "r_abs"),
                    _float(row, "scba_w_over_omega0"),
                )
            )
        axis.axvline(0.0, color="#888888", linestyle=":", linewidth=0.65)
        axis.set_xlim(*RESONANCE_SPECTRAL_WINDOW)
        axis.set_ylim(-0.08 * spacing, offsets[-1] + 1.08 * maximum)
        axis.set_yticks(offsets)
        axis.set_yticklabels(tick_labels, fontsize=7.6)
        for tick, color in zip(axis.get_yticklabels(), RESONANCE_CUT_COLORS):
            tick.set_color(color)
        axis.set_xlabel(r"$(E-E_B)/\omega_0$")
        axis.set_ylabel("measured sweep cut (vertical offset)")
        axis.set_title(r"$S=%g$: ED--SCBA spectral evolution" % s_value, pad=5.0)
        axis.grid(axis="x", alpha=0.12, linewidth=0.42)
        _panel_label(axis, chr(ord("c") + s_index))
    if spectral_legend is None:
        raise RuntimeError("The resonance-sweep plot has no spectral curves")
    fig.legend(
        handles=spectral_legend, loc="lower center", bbox_to_anchor=(0.5, 0.012),
        ncol=2, frameon=False, columnspacing=2.0,
    )
    fig.suptitle(
        r"Positive-$J$ evolution through the first odd commensurability",
        y=0.985, fontsize=10.5,
    )
    fig.subplots_adjust(left=0.135, right=0.99, bottom=0.095, top=0.865)
    paths = _save_figure(fig, output_dir, FIGURE_STEMS[1], dpi)
    return paths, sweep_rows, source_archives


def _write_line_scan_source(summary: Sequence[Dict[str, str]], path: Path) -> None:
    rows = [
        {key: row.get(key, "") for key in LINE_SCAN_SOURCE_COLUMNS}
        for row in summary
        if row.get("family") == "primary_fixed_S"
        and any(_close(_float(row, "huang_rhys"), value) for value in SELECTED_S)
    ]
    rows.sort(key=lambda row: (
        int(round(_float(row, "j_sign"))), _float(row, "huang_rhys"),
        _float(row, "r_abs"),
    ))
    _write_csv_atomic(path, rows, LINE_SCAN_SOURCE_COLUMNS)


def _manifest_row(
    stem: str, panels: str, outputs: Tuple[Path, Path], source: Path,
    selection: str, quantities: str, transformation: str, notes: str,
) -> Dict[str, Any]:
    png, pdf = outputs
    return {
        "figure_stem": stem, "panels": panels,
        "output_png": _portable_path(png), "output_pdf": _portable_path(pdf),
        "source_file": _portable_path(source), "source_sha256": _sha256(source),
        "source_size_bytes": source.stat().st_size, "selection": selection,
        "plotted_quantities": quantities, "transformation": transformation,
        "notes": notes,
    }


def _assert_exact_figure_set(output_dir: Path) -> None:
    expected = {
        stem + suffix for stem in FIGURE_STEMS for suffix in (".png", ".pdf")
    }
    actual = {
        path.name for path in output_dir.iterdir()
        if path.is_file() and path.suffix.lower() in (".png", ".pdf")
    }
    if not expected <= actual:
        raise RuntimeError(
            "Required command-16 figure files are missing: %s"
            % sorted(expected - actual)
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--dpi", type=int, default=320)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.dpi < 120:
        raise ValueError("--dpi must be at least 120")
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_style()
    summary, summary_by_id = _validate_inputs(results_dir)

    line_scan_source = results_dir / "wasserstein_line_scan_source.csv"
    peak_manifest = results_dir / "wasserstein_peak_manifest.csv"
    sweep_table = results_dir / "resonance_sweep_spectra_manifest.csv"
    _write_line_scan_source(summary, line_scan_source)
    peak_rows = extract_peak_manifest(summary)
    _write_csv_atomic(peak_manifest, peak_rows, PEAK_MANIFEST_COLUMNS)

    line_scan_outputs = plot_wasserstein_line_scan(summary, output_dir, args.dpi)
    sweep_outputs, sweep_rows, source_archives = plot_resonance_sweep_spectra(
        summary, summary_by_id, peak_rows, results_dir, output_dir, args.dpi
    )
    _write_csv_atomic(sweep_table, sweep_rows, RESONANCE_SWEEP_COLUMNS)

    manifest_rows: List[Dict[str, Any]] = []
    manifest_rows.append(_manifest_row(
        FIGURE_STEMS[0], "a--b", line_scan_outputs, line_scan_source,
        "primary_fixed_S; both J signs; S=0.02,0.05,0.10; measured r mesh",
        "w = W1(ED,SCBA)/omega0 only",
        "measured points joined in r order; no interpolation or smoothing",
        "Odd lines are allowed bare guides; even lines are parity-forbidden controls.",
    ))
    manifest_rows.append(_manifest_row(
        FIGURE_STEMS[0], "a--b", line_scan_outputs,
        results_dir / "local_baselines.csv",
        "primary J=0 rows; horizontal references for S=0.02,0.05,0.10 "
        "in both J-sign panels; all seven S values in inset",
        "w0(S) = W1(ED,SCBA)/omega0",
        "horizontal same-color lines for selected S; logarithmic inset for all S",
        "Dashed lines expose the local Franck--Condon background in both panels.",
    ))
    manifest_rows.append(_manifest_row(
        FIGURE_STEMS[0], "a--b", line_scan_outputs, peak_manifest,
        "integer windows n=1,...,5 for selected S and both J signs",
        "measured guide-window maxima and off-integer reference minima",
        "finite measured-point comparisons only",
        "A monotonic window is explicitly not labelled a resolved local cusp.",
    ))
    manifest_rows.append(_manifest_row(
        FIGURE_STEMS[1], "a--b", sweep_outputs, line_scan_source,
        "primary_fixed_S; J>0; S=0.05,0.10; 0.60<=r<=1.50 measured mesh",
        "w = W1(ED,SCBA)/omega0 and four measured cut indicators per S",
        "measured points joined in r order; no interpolation or smoothing",
        "The scan panels locate the lower flank, measured maximum, bare guide, and upper reference cut.",
    ))
    manifest_rows.append(_manifest_row(
        FIGURE_STEMS[1], "a--d", sweep_outputs, sweep_table,
        "four ordered measured cuts for each of S=0.05 and S=0.10",
        "case IDs, selection roles, r, w, J=0 baseline, areas, and archive paths",
        "none",
        "Compact provenance table linking every scan indicator to its stacked spectrum.",
    ))
    for row, archive_path in zip(sweep_rows, source_archives):
        manifest_rows.append(_manifest_row(
            FIGURE_STEMS[1], str(row["figure_panel"]), sweep_outputs, archive_path,
            str(row["case_id"]), "ED and SCBA bright absorption",
            "x=(E-E_B)/omega0 displayed on [-1,1.5]; each curve divided by its "
            "analysis-window raw area; "
            "y=omega0*A plus a common within-panel vertical offset",
            "%s (%s)" % (row["role"], row["selection_rule"]),
        ))
    manifest_path = output_dir / "figure_source_manifest.csv"
    _write_csv_atomic(manifest_path, manifest_rows, SOURCE_MANIFEST_COLUMNS)
    _assert_exact_figure_set(output_dir)

    print("Generated two active frequency-scan figure stems in %s" % output_dir)
    print("Resonance-sweep table: %s" % sweep_table)
    print("Figure source manifest: %s" % manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
