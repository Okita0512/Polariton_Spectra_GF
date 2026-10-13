#!/usr/bin/env python3
"""Plot the resolved-vibronic comparison used as main Fig. 4.

This is deliberately a plot-only pipeline.  It reads the validated S=2
archives shared with Supplementary Figs. S5 and S6 and never imports or calls
the ED/SCBA calculation drivers.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import numpy as np


S_VALUE = 2.0
ENERGY_00_EV = 2.0
PHONON_EV = 0.020
REORGANIZATION_EV = S_VALUE * PHONON_EV
VERTICAL_EV = ENERGY_00_EV + REORGANIZATION_EV
ETA_EV = 0.0005
SINGLE_MOLECULE_COUPLING_EV = 0.03536
TEMPERATURE_K = 0.0
X_LIMITS = (1.8, 2.3)
N_VALUES = (1, 2, 4, 8)
ROW_CLEARANCE = 0.08

FIGURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_DIR.parents[2]
CACHE_ROOT = PROJECT_ROOT / "Figures" / "shared" / "discrete_mode"
ZERO_ZERO_ROOT = CACHE_ROOT / "zero_zero"
VERTICAL_ROOT = CACHE_ROOT / "vertical"
OUTPUT = FIGURE_DIR / "Fig4.pdf"
VALIDATION = FIGURE_DIR / "figure_validation.txt"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    source_anchor: str
    path: Path
    tuning: str
    n_molecules: int | None
    cavity_frequency_ev: float | None
    expected_sha256: str


CASES = (
    CaseSpec(
        "outside",
        "S5(c) [identical spectral arrays to S6(c)]",
        ZERO_ZERO_ROOT / "results" / "outside_S2.0.npz",
        "outside",
        None,
        None,
        "36422ee10b78b158068c6d62058bd4c170e5f520da42a8babaa6f0b4e2af2dbf",
    ),
    CaseSpec(
        "zero_zero_N1",
        "S5(f)",
        ZERO_ZERO_ROOT / "results" / "N1_S2.0.npz",
        "zero_zero",
        1,
        ENERGY_00_EV,
        "29241a112638360e6b71e152724b4899e9993b35f571ffd976d92944ac465a30",
    ),
    CaseSpec(
        "zero_zero_N2",
        "S5(i)",
        ZERO_ZERO_ROOT / "results" / "N2_S2.0.npz",
        "zero_zero",
        2,
        ENERGY_00_EV,
        "1ca39d2dd8f17598270afda15fe3a2543f3e0af38338b34c58743527fa444f79",
    ),
    CaseSpec(
        "zero_zero_N4",
        "S5(l)",
        ZERO_ZERO_ROOT / "results" / "N4_S2.0.npz",
        "zero_zero",
        4,
        ENERGY_00_EV,
        "ce4635d08a44ad53fefcc862edb2b29c4c527735b73d7217c7565622f1f4e922",
    ),
    CaseSpec(
        "zero_zero_N8",
        "S5(o)",
        ZERO_ZERO_ROOT / "results" / "N8_S2.0.npz",
        "zero_zero",
        8,
        ENERGY_00_EV,
        "3f4a36c867bc2f028ef4b0dc2be144dd465178311e7afe1f83af626a31c1b894",
    ),
    CaseSpec(
        "vertical_N1",
        "S6(f)",
        VERTICAL_ROOT / "results" / "N1_S2.0.npz",
        "vertical",
        1,
        VERTICAL_EV,
        "eee596cc52ad3e4fb87bf4b16f927a887da63618246cc68d1bc6ae9604d6e7dd",
    ),
    CaseSpec(
        "vertical_N2",
        "S6(i)",
        VERTICAL_ROOT / "results" / "N2_S2.0.npz",
        "vertical",
        2,
        VERTICAL_EV,
        "2fffbd00f1765fdfe735ae16b2301edd14c58553383661aa52849bd4929ea357",
    ),
    CaseSpec(
        "vertical_N4",
        "S6(l)",
        VERTICAL_ROOT / "results" / "N4_S2.0.npz",
        "vertical",
        4,
        VERTICAL_EV,
        "25ea12e9b729626e6fa7e15b0bae676f32205d4bd1cf2e42f022960c28c5cbe8",
    ),
    CaseSpec(
        "vertical_N8",
        "S6(o)",
        VERTICAL_ROOT / "results" / "N8_S2.0.npz",
        "vertical",
        8,
        VERTICAL_EV,
        "f50b019e6c16079f15c175699692a4f768f18ff9fd0d2af21f55c51387864bef",
    ),
)


def configure_fonts() -> None:
    """Use regular-weight Helvetica uniformly and unsimplified paths."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
            "font.weight": "normal",
            "font.size": 7.8,
            "axes.labelsize": 8.2,
            "axes.labelweight": "normal",
            "axes.titlesize": 8.6,
            "axes.titleweight": "normal",
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.6,
            "pdf.use14corefonts": True,
            "ps.useafm": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.linewidth": 0.65,
            "path.simplify": False,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_close(actual: float, expected: float, name: str, path: Path) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"{name}={actual!r}, expected {expected!r}, in {path}")


def require_calculation_validation(path: Path, expected_heading: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing calculation validation: {path}")
    text = path.read_text(encoding="utf-8")
    for required in (expected_heading, "status = PASS", "T=0"):
        if required not in text:
            raise RuntimeError(f"Missing {required!r} in {path}")


def load_case(spec: CaseSpec) -> dict[str, object]:
    """Load one raw cache and reject stale, modified, or misassigned input."""

    if not spec.path.is_file():
        raise FileNotFoundError(f"Missing cached spectrum: {spec.path}")
    source_hash = sha256(spec.path)
    if source_hash != spec.expected_sha256:
        raise ValueError(
            f"Cache hash changed for {spec.path}: {source_hash}; "
            f"expected {spec.expected_sha256}"
        )
    with np.load(spec.path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        result: dict[str, object] = {
            "spec": spec,
            "source_hash": source_hash,
            "metadata": metadata,
            "exact_energy": archive["exact_energy_eV"].copy(),
            "exact_spectrum": archive["exact_spectrum_per_eV"].copy(),
            "scba_energy": archive["scba_energy_eV"].copy(),
            "scba_spectrum": archive["scba_spectrum_per_eV"].copy(),
        }

    model = metadata["model"]
    numerics = metadata["numerics"]
    require_close(float(model["huang_rhys"]), S_VALUE, "S", spec.path)
    require_close(float(model["energy_00"]), ENERGY_00_EV, "omega_00", spec.path)
    require_close(
        float(model["vertical_exciton_energy"]), VERTICAL_EV, "omega_0", spec.path
    )
    require_close(float(model["phonon_frequency"]), PHONON_EV, "nu_0", spec.path)
    require_close(float(model["eta"]), ETA_EV, "eta", spec.path)
    require_close(float(numerics["plot_min"]), X_LIMITS[0], "plot_min", spec.path)
    require_close(float(numerics["plot_max"]), X_LIMITS[1], "plot_max", spec.path)
    if int(numerics["ed_lanczos_steps"]) != 5000:
        raise ValueError(f"Lanczos-production metadata mismatch in {spec.path}")
    if int(numerics["ed_lanczos_check_steps"]) != 4000:
        raise ValueError(f"Lanczos-check metadata mismatch in {spec.path}")

    inside = spec.n_molecules is not None
    if bool(model["inside_cavity"]) is not inside:
        raise ValueError(f"Inside/outside assignment mismatch in {spec.path}")
    if inside:
        assert spec.n_molecules is not None
        assert spec.cavity_frequency_ev is not None
        if int(model["n_molecules"]) != spec.n_molecules:
            raise ValueError(f"N assignment mismatch in {spec.path}")
        require_close(
            float(model["cavity_frequency"]),
            spec.cavity_frequency_ev,
            "omega_c",
            spec.path,
        )
        require_close(
            float(model["single_molecule_coupling"]),
            SINGLE_MOLECULE_COUPLING_EV,
            "g_c",
            spec.path,
        )
        require_close(
            float(model["collective_coupling"]),
            math.sqrt(spec.n_molecules) * SINGLE_MOLECULE_COUPLING_EV,
            "sqrt(N) g_c",
            spec.path,
        )
        expected_method = (
            "dense_eigh" if spec.n_molecules in (1, 2) else "projected_lanczos_cf"
        )
        if metadata["exact_method"] != expected_method:
            raise ValueError(f"Unexpected exact method in {spec.path}")
    else:
        require_close(float(model["single_molecule_coupling"]), 0.0, "g_c", spec.path)
        if metadata["exact_method"] != "analytic_franck_condon":
            raise ValueError(f"Outside exact reference is not analytic in {spec.path}")

    for name in ("exact_energy", "exact_spectrum", "scba_energy", "scba_spectrum"):
        values = np.asarray(result[name], dtype=float)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} is not a finite 1D array in {spec.path}")
    return result


def validate_common_outside_reference() -> None:
    """Verify that the two supporting grids contain the same outside curves."""

    zero_path = ZERO_ZERO_ROOT / "results" / "outside_S2.0.npz"
    vertical_path = VERTICAL_ROOT / "results" / "outside_S2.0.npz"
    keys = (
        "exact_energy_eV",
        "exact_spectrum_per_eV",
        "scba_energy_eV",
        "scba_spectrum_per_eV",
    )
    with np.load(zero_path, allow_pickle=False) as zero, np.load(
        vertical_path, allow_pickle=False
    ) as vertical:
        for key in keys:
            if not np.array_equal(zero[key], vertical[key]):
                raise ValueError(f"S5(c) and S6(c) differ for cached array {key}")


def display_pair(case: dict[str, object]) -> dict[str, np.ndarray | float]:
    """Apply exactly one peak/area display transform over 1.8--2.3 eV."""

    exact_energy = np.asarray(case["exact_energy"], dtype=float)
    exact_raw = np.asarray(case["exact_spectrum"], dtype=float)
    scba_energy = np.asarray(case["scba_energy"], dtype=float)
    scba_raw = np.asarray(case["scba_spectrum"], dtype=float)
    exact_mask = (exact_energy >= X_LIMITS[0]) & (exact_energy <= X_LIMITS[1])
    scba_mask = (scba_energy >= X_LIMITS[0]) & (scba_energy <= X_LIMITS[1])
    if np.count_nonzero(exact_mask) < 2 or np.count_nonzero(scba_mask) < 2:
        raise ValueError("Spectrum does not cover the displayed interval")

    exact_energy_display = exact_energy[exact_mask]
    exact_raw_display = exact_raw[exact_mask]
    scba_energy_display = scba_energy[scba_mask]
    scba_raw_display = scba_raw[scba_mask]
    exact_peak = float(np.max(exact_raw_display))
    exact_area = float(np.trapz(exact_raw_display, exact_energy_display))
    scba_area = float(np.trapz(scba_raw_display, scba_energy_display))
    if exact_peak <= 0.0 or exact_area <= 0.0 or scba_area <= 0.0:
        raise ValueError("Display normalization requires positive spectra")

    scba_area_factor = exact_area / scba_area
    exact_display_factor = 1.0 / exact_peak
    scba_display_factor = scba_area_factor / exact_peak
    exact_display = exact_raw_display * exact_display_factor
    scba_display = scba_raw_display * scba_display_factor
    matched_area = float(np.trapz(scba_display, scba_energy_display))
    reference_area = float(np.trapz(exact_display, exact_energy_display))
    if not np.isclose(matched_area, reference_area, rtol=2.0e-12, atol=1.0e-12):
        raise ValueError("SCBA curve failed the displayed-interval area match")

    diagnostics = case["metadata"]["diagnostics"]
    require_close(
        float(diagnostics["scba_area_scale"]),
        scba_area_factor,
        "cached scba_area_scale",
        case["spec"].path,
    )
    return {
        "exact_energy": exact_energy_display,
        "exact": exact_display,
        "scba_energy": scba_energy_display,
        "scba": scba_display,
        "exact_peak": exact_peak,
        "exact_area": exact_area,
        "scba_area": scba_area,
        "exact_display_factor": exact_display_factor,
        "scba_area_factor": scba_area_factor,
        "scba_display_factor": scba_display_factor,
        "area_ratio_after_matching": matched_area / reference_area,
        "display_max": max(float(np.max(exact_display)), float(np.max(scba_display))),
    }


def style_axis(axis: plt.Axes) -> None:
    axis.set_xlim(*X_LIMITS)
    axis.xaxis.set_major_locator(MultipleLocator(0.1))
    axis.xaxis.set_minor_locator(MultipleLocator(0.02))
    axis.tick_params(which="both", direction="in", top=True, right=True, width=0.6)
    axis.tick_params(which="major", length=3.2)
    axis.tick_params(which="minor", length=1.8)


def plot_pair(axis: plt.Axes, pair: dict[str, np.ndarray | float], offset: float) -> int:
    axis.plot(
        np.asarray(pair["exact_energy"]),
        np.asarray(pair["exact"]) + offset,
        color="black",
        linewidth=0.82,
        solid_joinstyle="round",
        zorder=3,
    )
    axis.plot(
        np.asarray(pair["scba_energy"]),
        np.asarray(pair["scba"]) + offset,
        color="#D62728",
        linestyle=(0, (4.0, 2.0)),
        linewidth=0.82,
        dash_capstyle="butt",
        zorder=2,
    )
    return 2


def plot_figure() -> tuple[Path, Path]:
    """Generate the vector asset and its auditable validation record."""

    require_calculation_validation(
        ZERO_ZERO_ROOT / "calculation_validation.txt",
        "FigS5 fixed-g_c ED/SCBA validation",
    )
    require_calculation_validation(
        VERTICAL_ROOT / "calculation_validation.txt",
        "FigS6 vertical-resonance ED/SCBA validation",
    )
    validate_common_outside_reference()
    loaded = {spec.key: load_case(spec) for spec in CASES}
    displayed = {key: display_pair(case) for key, case in loaded.items()}

    largest_pair_height = max(
        float(displayed[spec.key]["display_max"]) for spec in CASES
    )
    largest_cavity_pair_height = max(
        float(displayed[spec.key]["display_max"])
        for spec in CASES
        if spec.n_molecules is not None
    )
    row_step = (
        math.ceil((largest_cavity_pair_height + ROW_CLEARANCE) * 100.0) / 100.0
    )
    offsets = {n: (len(N_VALUES) - 1 - index) * row_step for index, n in enumerate(N_VALUES)}
    outside_offset = len(N_VALUES) * row_step
    top_extent = outside_offset + float(displayed["outside"]["display_max"])

    fig = plt.figure(figsize=(7.08, 3.55))
    grid = fig.add_gridspec(
        1,
        2,
        left=0.075,
        right=0.988,
        bottom=0.115,
        top=0.955,
        wspace=0.13,
    )
    zero_axis = fig.add_subplot(grid[0, 0])
    vertical_axis = fig.add_subplot(grid[0, 1], sharex=zero_axis)
    curve_count = 0

    for axis, prefix, panel, title in (
        (zero_axis, "zero_zero", "a", "0\N{EN DASH}0 tuning"),
        (vertical_axis, "vertical", "b", "Vertical tuning"),
    ):
        axis.axhline(
            outside_offset,
            color="0.74",
            linewidth=0.42,
            linestyle=(0, (1.3, 2.1)),
            zorder=0,
        )
        curve_count += plot_pair(axis, displayed["outside"], outside_offset)
        axis.text(
            0.982,
            outside_offset + 0.075,
            "Outside cavity",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7.2,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7, "alpha": 0.88},
            zorder=5,
        )
        for n_value in N_VALUES:
            offset = offsets[n_value]
            axis.axhline(
                offset,
                color="0.74",
                linewidth=0.42,
                linestyle=(0, (1.3, 2.1)) if offset else "-",
                zorder=0,
            )
            curve_count += plot_pair(axis, displayed[f"{prefix}_N{n_value}"], offset)
            axis.text(
                0.982,
                offset + 0.075,
                f"N = {n_value}",
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=7.2,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7, "alpha": 0.88},
                zorder=5,
            )
        axis.set_ylim(-0.04, top_extent + 0.13)
        axis.set_yticks(())
        axis.set_xlabel("Energy (eV)")
        axis.set_title(f"({panel})  {title}", loc="left", pad=3.0)

    zero_axis.set_ylabel("Normalized intensity (offset)", labelpad=6.0)
    for axis in (zero_axis, vertical_axis):
        style_axis(axis)

    legend_handles = (
        Line2D([], [], color="black", linewidth=0.95, label="Exact"),
        Line2D(
            [],
            [],
            color="#D62728",
            linestyle=(0, (4.0, 2.0)),
            linewidth=0.95,
            label="SCBA",
        ),
    )
    zero_axis.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=1,
        frameon=False,
        borderaxespad=0.25,
        handlelength=2.7,
        columnspacing=1.5,
    )

    if len(CASES) != 9 or curve_count != 20:
        raise RuntimeError(
            f"Expected 9 unique cases/20 displayed curves, got "
            f"{len(CASES)}/{curve_count}"
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT,
        metadata={
            "Title": "Resolved vibronic spectra at S=2",
            "Subject": "Main Figure 4: two tuning panels with a shared outside reference",
            "Creator": "Figures/Main/Fig4/plot_Fig4.py",
        },
    )
    plt.close(fig)

    audit_lines = [
        "Main Fig. 4 resolved-vibronic validation",
        "========================================",
        "status = PASS",
        "output = Figures/Main/Fig4/Fig4.pdf",
        "format = vector PDF; Matplotlib line/text artists only; path simplification disabled",
        "layout = two equal tuning panels; outside-cavity row repeated above N=1,2,4,8",
        "font = regular-weight Helvetica throughout, including panel labels and legend",
        "legend = Exact (black solid); SCBA (red dashed)",
        "unique case count = 9",
        "displayed row count = 10 (five per panel; outside reference repeated)",
        "scientific curve count = 20 displayed traces (10 exact/SCBA pairs)",
        "panel (a) = outside reference then 0-0 tuning S5(f,i,l,o), N=1,2,4,8",
        "panel (b) = outside reference then vertical tuning S6(f,i,l,o), N=1,2,4,8",
        "outside exact method = analytic Franck-Condon progression",
        "cavity exact methods = dense ED (N=1,2); projected Lanczos ED (N=4,8)",
        "energy interval = 1.8--2.3 eV; full cached samples in that interval are plotted",
        "S = 2; nu_0 = 0.020 eV; lambda = 0.040 eV",
        "omega_00 = 2.000 eV; omega_0 = 2.040 eV; T = 0",
        "eta = 0.0005 eV common retarded Lorentzian half-width (not kappa or vibrational damping)",
        "g_c = 0 outside; fixed single-molecule g_c = 0.03536 eV in cavity cases",
        "omega_c = omega_00 in (a); omega_c = omega_0 in (b)",
        "normalization = E/max(E); C*[integral(E)/integral(C)]/max(E), separately per pair over 1.8--2.3 eV",
        "SCBA is not independently peak-normalized; each pair receives one common vertical offset",
        f"largest exact/SCBA display height = {largest_pair_height:.15e}",
        f"largest cavity exact/SCBA display height = {largest_cavity_pair_height:.15e}",
        f"row clearance target = {ROW_CLEARANCE:.6f}",
        f"common row step derived from cavity pairs in both protocols = {row_step:.6f}",
        f"outside offset = {outside_offset:.6f}",
        "offsets (N=1,2,4,8) = " + ", ".join(f"{offsets[n]:.6f}" for n in N_VALUES),
        "S5(c) versus S6(c) spectral arrays = exactly identical",
        "calculation validations for both shared grids = PASS",
        "cached arrays modified by plot = no",
        "",
        "case,anchor,source,sha256,N,exact_method,exact_peak,exact_display_factor,scba_area_factor,scba_display_factor,matched_area_ratio,display_max,offset",
    ]
    for spec in CASES:
        pair = displayed[spec.key]
        exact_method = loaded[spec.key]["metadata"]["exact_method"]
        offset = outside_offset if spec.n_molecules is None else offsets[spec.n_molecules]
        audit_lines.append(
            ",".join(
                (
                    spec.key,
                    spec.source_anchor,
                    str(spec.path.relative_to(PROJECT_ROOT)),
                    str(loaded[spec.key]["source_hash"]),
                    "outside" if spec.n_molecules is None else str(spec.n_molecules),
                    str(exact_method),
                    f"{float(pair['exact_peak']):.15e}",
                    f"{float(pair['exact_display_factor']):.15e}",
                    f"{float(pair['scba_area_factor']):.15e}",
                    f"{float(pair['scba_display_factor']):.15e}",
                    f"{float(pair['area_ratio_after_matching']):.15e}",
                    f"{float(pair['display_max']):.15e}",
                    f"{offset:.6f}",
                )
            )
        )
    audit_lines.append("")
    VALIDATION.write_text("\n".join(audit_lines), encoding="utf-8")
    return OUTPUT, VALIDATION


def main() -> None:
    configure_fonts()
    for output in plot_figure():
        print(f"Saved {output.relative_to(PROJECT_ROOT)}")
    print("Figure validation: PASS")


if __name__ == "__main__":
    main()
