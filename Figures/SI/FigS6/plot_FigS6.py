#!/usr/bin/env python3
"""Generate final SI Fig. S6: dimer scans plus two actual spectra.

The upper row is the complete calculated spectral-distance scan for both
signs of J.  The lower row contains unit-area ED and SCBA spectra at the two
marked positive-J points.  Curves are drawn directly from calculated arrays;
there is no interpolation, smoothing, filtering, or downsampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


FIGURE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FIGURE_DIR.parents[2]
REVISION27_DIR = PROJECT_ROOT / "Figures/common/dimer/single_mode/revision27"
for candidate in (PROJECT_ROOT, REVISION27_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from Figures.common.dimer.single_mode import workflow as sm  # noqa: E402
from Figures.common.si_plot_style import (  # noqa: E402
    EXACT_STYLE,
    SCBA_STYLE,
    S_COLORS,
    add_panel_label,
    compact_legend_kwargs,
    configure_si_matplotlib,
    style_axis,
)
import audit_dimer_revision27 as r27  # noqa: E402


SELECTED_S = (0.02, 0.05, 0.10)
SIGNS = (+1, -1)
BASELINE_DASH = (0, (5.0, 2.2))
SPECTRUM_SUPPORT = (-18.0, 18.0)
SPECTRUM_DISPLAY = (-1.25, 1.55)
ARRAY_COLUMNS = ("x", "ed_probability_density", "scba_probability_density")
PROVENANCE_COLUMNS = (
    "panel", "case_id", "source_kind", "source_path", "source_sha256",
    "huang_rhys", "j_sign", "j_meV", "r_abs", "nu0_meV", "eta_meV",
    "eta_over_nu0", "dx", "n_vib", "normalization_support_x",
    "display_window_x", "ed_raw_area_over_E", "scba_raw_area_over_E",
    "ed_unit_area_over_x", "scba_unit_area_over_x", "w_from_spectrum",
    "w_from_scan", "absolute_w_difference", "array_path",
)
MANIFEST_COLUMNS = ("kind", "path", "sha256", "size_bytes", "role")


def _number(row: Mapping[str, Any], key: str) -> float:
    return float(str(row.get(key, "nan")))


def _scan_row(
    mesh_rows: Sequence[Mapping[str, Any]], s_value: float, sign: int, r_value: float
) -> Mapping[str, Any]:
    matches = [
        row for row in mesh_rows
        if abs(_number(row, "huang_rhys") - s_value) <= 1.0e-12
        and int(round(_number(row, "j_sign"))) == sign
        and abs(_number(row, "r_abs") - r_value) <= 1.0e-12
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one scan row for S={s_value:g}, sign={sign:+d}, "
            f"r={r_value:.3f}; found {len(matches)}"
        )
    return matches[0]


def _unit_area_from_archive(
    cut_rows: Sequence[Mapping[str, Any]], mesh_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Retrieve the validated production r=0.800 spectrum archive."""

    matches = [
        row for row in cut_rows
        if str(row.get("case_id", "")) == "eta005_S0p1_Jp_r0p8"
    ]
    if len(matches) != 1:
        raise ValueError("The validated S=0.10, r=0.800 archive is unavailable")
    row = matches[0]
    energy = np.asarray(row["_energy_eV"], dtype=float)
    mask = np.asarray(row["_analysis_mask"], dtype=bool)
    ed_raw = np.asarray(row["_ed_spectrum"], dtype=float)[mask]
    scba_raw = np.asarray(row["_scba_spectrum"], dtype=float)[mask]
    energy = energy[mask]
    metadata = row["_metadata"]
    model = metadata["model"]
    nu0_eV = float(model["omega0_meV"]) / 1000.0
    bright_energy = (
        float(model["energy_00_eV"])
        + float(model["huang_rhys"]) * nu0_eV
        + float(model["j_meV"]) / 1000.0
    )
    x = (energy - bright_energy) / nu0_eV
    ed_area_e = float(np.trapz(ed_raw, energy))
    scba_area_e = float(np.trapz(scba_raw, energy))
    # dE = nu0 dx, hence p_x = nu0 A(E) / integral A(E)dE.
    ed_probability = nu0_eV * ed_raw / ed_area_e
    scba_probability = nu0_eV * scba_raw / scba_area_e
    # Remove only floating-point roundoff in the archive's complex64 arrays.
    ed_probability /= float(np.trapz(ed_probability, x))
    scba_probability /= float(np.trapz(scba_probability, x))
    _cdf, w_value = r27._metric_from_probabilities(  # noqa: SLF001
        x, ed_probability, scba_probability
    )
    scan = _scan_row(mesh_rows, 0.10, +1, 0.800)
    source_path = PROJECT_ROOT / str(row["archive_file"])
    return {
        "panel": "c",
        "case_id": str(row["case_id"]),
        "source_kind": "validated production spectrum archive",
        "source_path": source_path,
        "source_sha256": sm.sha256(source_path),
        "huang_rhys": 0.10,
        "j_sign": +1,
        "j_meV": 25.0,
        "r_abs": 0.800,
        "nu0_meV": float(model["omega0_meV"]),
        "eta_meV": float(model["eta_meV"]),
        "eta_over_nu0": float(model["eta_meV"]) / float(model["omega0_meV"]),
        "dx": float(metadata["dimensionless_grid"]["dx"]),
        "n_vib": int(metadata["n_vib"]),
        "x": x,
        "ed_probability": ed_probability,
        "scba_probability": scba_probability,
        "ed_raw_area_over_E": ed_area_e,
        "scba_raw_area_over_E": scba_area_e,
        "w_from_spectrum": w_value,
        "w_from_scan": _number(scan, "w_W1_over_nu0"),
    }


def _unit_area_from_solver(mesh_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Regenerate the missing production r=1.880 spectral arrays."""

    solution = r27._solve(  # noqa: SLF001
        "command28_S010_Jp_r1880_production",
        0.10,
        +1,
        1.880,
        dx=0.002,
        phase=0.0,
        n_vib=10,
    )
    scan = _scan_row(mesh_rows, 0.10, +1, 1.880)
    driver_path = REVISION27_DIR / "audit_dimer_revision27.py"
    return {
        "panel": "d",
        "case_id": solution.label,
        "source_kind": "regenerated with verified revision-27 production solver",
        "source_path": driver_path,
        "source_sha256": sm.sha256(driver_path),
        "huang_rhys": solution.s_value,
        "j_sign": solution.sign,
        "j_meV": 25.0,
        "r_abs": solution.r_value,
        "nu0_meV": 50.0 / solution.r_value,
        "eta_meV": 0.005 * 50.0 / solution.r_value,
        "eta_over_nu0": 0.005,
        "dx": solution.dx,
        "n_vib": solution.n_vib,
        "x": solution.x,
        "ed_probability": solution.ed_probability,
        "scba_probability": solution.scba_probability,
        "ed_raw_area_over_E": solution.ed_area,
        "scba_raw_area_over_E": solution.scba_area,
        "w_from_spectrum": solution.w,
        "w_from_scan": _number(scan, "w_W1_over_nu0"),
    }


def _validate_spectrum(case: Mapping[str, Any]) -> None:
    x = np.asarray(case["x"], dtype=float)
    ed = np.asarray(case["ed_probability"], dtype=float)
    scba = np.asarray(case["scba_probability"], dtype=float)
    if (
        x.ndim != 1
        or ed.shape != x.shape
        or scba.shape != x.shape
        or x.size != 18001
        or not np.all(np.diff(x) > 0.0)
        or abs(x[0] - SPECTRUM_SUPPORT[0]) > 2.0e-10
        or abs(x[-1] - SPECTRUM_SUPPORT[1]) > 2.0e-10
        or abs(float(np.median(np.diff(x))) - 0.002) > 2.0e-12
        or np.min(ed) < -1.0e-12
        or np.min(scba) < -1.0e-12
        or abs(float(np.trapz(ed, x)) - 1.0) > 2.0e-12
        or abs(float(np.trapz(scba, x)) - 1.0) > 2.0e-12
        or abs(float(case["w_from_spectrum"]) - float(case["w_from_scan"]))
        > 2.0e-9
    ):
        raise ValueError(f"Spectrum validation failed for panel {case['panel']}")


def _write_spectrum_products(cases: Sequence[Mapping[str, Any]]) -> tuple[list[Path], Path]:
    processed = FIGURE_DIR / "processed"
    array_paths: list[Path] = []
    provenance_rows: list[dict[str, Any]] = []
    for case in cases:
        r_token = f"{float(case['r_abs']):.3f}".replace(".", "p")
        array_path = processed / f"spectrum_S0p10_Jp_r{r_token}.csv"
        rows = [
            {
                "x": float(x_value),
                "ed_probability_density": float(ed_value),
                "scba_probability_density": float(scba_value),
            }
            for x_value, ed_value, scba_value in zip(
                case["x"], case["ed_probability"], case["scba_probability"]
            )
        ]
        sm.write_csv(array_path, rows, ARRAY_COLUMNS)
        array_paths.append(array_path)
        provenance_rows.append({
            "panel": case["panel"],
            "case_id": case["case_id"],
            "source_kind": case["source_kind"],
            "source_path": sm.project_relative(Path(case["source_path"]), PROJECT_ROOT),
            "source_sha256": case["source_sha256"],
            "huang_rhys": case["huang_rhys"],
            "j_sign": case["j_sign"],
            "j_meV": case["j_meV"],
            "r_abs": case["r_abs"],
            "nu0_meV": case["nu0_meV"],
            "eta_meV": case["eta_meV"],
            "eta_over_nu0": case["eta_over_nu0"],
            "dx": case["dx"],
            "n_vib": case["n_vib"],
            "normalization_support_x": "[-18,18]",
            "display_window_x": "[-1.25,1.55]",
            "ed_raw_area_over_E": case["ed_raw_area_over_E"],
            "scba_raw_area_over_E": case["scba_raw_area_over_E"],
            "ed_unit_area_over_x": float(np.trapz(case["ed_probability"], case["x"])),
            "scba_unit_area_over_x": float(np.trapz(case["scba_probability"], case["x"])),
            "w_from_spectrum": case["w_from_spectrum"],
            "w_from_scan": case["w_from_scan"],
            "absolute_w_difference": abs(
                float(case["w_from_spectrum"]) - float(case["w_from_scan"])
            ),
            "array_path": sm.project_relative(array_path, PROJECT_ROOT),
        })
    provenance_path = processed / "spectral_provenance.csv"
    sm.write_csv(provenance_path, provenance_rows, PROVENANCE_COLUMNS)
    return array_paths, provenance_path


def _above_axes_title(axis: plt.Axes, title: str) -> None:
    axis.annotate(
        title,
        xy=(1.0, 1.0),
        xycoords="axes fraction",
        xytext=(0.0, 4.0),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9.0,
        annotation_clip=False,
        clip_on=False,
    )


def _plot(
    mesh_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    spectrum_cases: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    configure_si_matplotlib()
    fig, axes = plt.subplots(
        2, 2, figsize=(7.20, 5.55), sharex="row", sharey="row"
    )
    scan_axes = axes[0]
    spectrum_axes = axes[1]
    baseline_by_s = {
        round(_number(row, "huang_rhys"), 12): _number(row, "w0_W1_over_nu0")
        for row in baseline_rows
    }

    for panel_index, (axis, sign) in enumerate(zip(scan_axes, SIGNS)):
        for s_value in SELECTED_S:
            group = sorted(
                (
                    row for row in mesh_rows
                    if abs(_number(row, "huang_rhys") - s_value) <= 1.0e-12
                    and int(round(_number(row, "j_sign"))) == sign
                ),
                key=lambda row: _number(row, "r_abs"),
            )
            axis.plot(
                [_number(row, "r_abs") for row in group],
                [_number(row, "w_W1_over_nu0") for row in group],
                color=S_COLORS[s_value], linewidth=0.95, zorder=4,
            )
            axis.axhline(
                baseline_by_s[round(s_value, 12)],
                color=S_COLORS[s_value], linestyle=BASELINE_DASH,
                linewidth=0.70, alpha=0.84, zorder=1,
            )
        axis.set_xlim(0.25, 5.5)
        axis.set_ylim(0.0, 0.0535)
        axis.set_xticks((0.5, 1, 2, 3, 4, 5, 5.5))
        axis.set_yticks((0.00, 0.01, 0.02, 0.03, 0.04, 0.05))
        axis.set_xlabel(r"$r=2|J|/\nu_0$")
        style_axis(axis)
        add_panel_label(axis, f"({chr(ord('a') + panel_index)})")
        _above_axes_title(axis, rf"$J={'+' if sign > 0 else '-'}25\ \mathrm{{meV}}$")
    scan_axes[0].set_ylabel(r"$w=W_1/\nu_0$")

    for r_value, marker, label in ((0.800, "o", "c"), (1.880, "s", "d")):
        row = _scan_row(mesh_rows, 0.10, +1, r_value)
        w_value = _number(row, "w_W1_over_nu0")
        scan_axes[0].scatter(
            [r_value], [w_value], marker=marker, s=29.0,
            facecolor="white", edgecolor="black", linewidth=0.80, zorder=8,
        )
        scan_axes[0].annotate(
            label, (r_value, w_value), xytext=(0, 5),
            textcoords="offset points", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color="black", zorder=9,
        )

    scan_handles = [
        Line2D([0], [0], color=S_COLORS[s_value], linewidth=1.30,
               label=rf"$S={s_value:.2f}$")
        for s_value in SELECTED_S
    ] + [
        Line2D([0], [0], color="#666666", linestyle=BASELINE_DASH,
               linewidth=0.78, label=r"$J=0$ baselines")
    ]
    scan_axes[1].legend(
        handles=scan_handles, loc="upper right", ncol=2,
        **compact_legend_kwargs(),
    )

    spectrum_handles: tuple[Line2D, Line2D] | None = None
    for axis, case in zip(spectrum_axes, spectrum_cases):
        x = np.asarray(case["x"], dtype=float)
        display = (
            (x >= SPECTRUM_DISPLAY[0] - 1.0e-12)
            & (x <= SPECTRUM_DISPLAY[1] + 1.0e-12)
        )
        exact_line, = axis.plot(
            x[display], np.asarray(case["ed_probability"])[display],
            label="Exact", **EXACT_STYLE,
        )
        scba_line, = axis.plot(
            x[display], np.asarray(case["scba_probability"])[display],
            label="SCBA", **SCBA_STYLE,
        )
        spectrum_handles = (exact_line, scba_line)
        axis.set_xlim(*SPECTRUM_DISPLAY)
        axis.set_ylim(0.0, 61.5)
        axis.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0, 1.5))
        axis.set_yticks((0, 15, 30, 45, 60))
        axis.set_xlabel(r"$x=(E-E_B)/\nu_0$")
        style_axis(axis)
        add_panel_label(axis, f"({case['panel']})")
        _above_axes_title(
            axis,
            rf"$S=0.10$, $J=+25\ \mathrm{{meV}}$, "
            rf"$r={float(case['r_abs']):.3f}$",
        )
    spectrum_axes[0].set_ylabel(r"Normalized density $p_x(x)$")
    if spectrum_handles is None:
        raise RuntimeError("No spectral panels were drawn")
    spectrum_axes[0].legend(
        handles=spectrum_handles, loc="upper right", ncol=1,
        **compact_legend_kwargs(),
    )

    fig.subplots_adjust(
        left=0.090, right=0.992, bottom=0.090, top=0.955,
        wspace=0.080, hspace=0.355,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, format="pdf", bbox_inches="tight", pad_inches=0.025,
        metadata={
            "Title": "SI Fig. S6: dimer scans and selected spectra",
            "Author": "single-mode dimer validation workflow",
            "Subject": "Actual calculated arrays; no smoothing or interpolation",
            "Creator": "Figures/SI/FigS6/plot_FigS6.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)


def _validation_text(
    mesh_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "Final Fig. S6 merged dimer validation",
        "=====================================",
        "status = PASS",
        f"scan rows = {len(mesh_rows)} calculated points",
        "scan composition = 2214 production points + 34 revision-27 calculated refined-r points",
        "selected S = 0.02, 0.05, 0.10",
        "J = +/-25 meV; T = 0 K; eta/nu0 = 0.005",
        "scan r range = 0.25 .. 5.5",
        "scan construction = calculated points joined in increasing r",
        "smoothing/interpolation/filtering/downsampling = none",
        "spectrum normalization support = x in [-18,18]",
        "spectrum display window = x in [-1.25,1.55]",
        "Jacobian = dE=nu0 dx; p_x=nu0 A(E)/integral_[support] A(E)dE",
        "spectral grid = Delta x=0.002; 18001 support samples; N_vib=10",
        "r=0.800 spectrum = retrieved validated production archive",
        "r=1.880 spectrum = regenerated with the verified revision-27 solver at production settings",
        "", "J=0 baselines", "--------------",
    ]
    for s_value in SELECTED_S:
        row = next(
            row for row in baseline_rows
            if abs(_number(row, "huang_rhys") - s_value) <= 1.0e-12
        )
        lines.append(f"S={s_value:g}: w0={_number(row, 'w0_W1_over_nu0'):.15g}")
    lines.extend(["", "Spectral-panel values", "---------------------"])
    for case in cases:
        lines.extend([
            (
                f"panel ({case['panel']}): S={float(case['huang_rhys']):.2f}, "
                f"J={float(case['j_meV']):+.0f} meV, r={float(case['r_abs']):.3f}, "
                f"nu0={float(case['nu0_meV']):.15g} meV, "
                f"eta={float(case['eta_meV']):.15g} meV"
            ),
            f"  source = {sm.project_relative(Path(case['source_path']), PROJECT_ROOT)}",
            f"  source sha256 = {case['source_sha256']}",
            (
                f"  raw areas over E: Exact={float(case['ed_raw_area_over_E']):.15g}; "
                f"SCBA={float(case['scba_raw_area_over_E']):.15g}"
            ),
            (
                f"  unit areas over x: Exact={float(np.trapz(case['ed_probability'], case['x'])):.15g}; "
                f"SCBA={float(np.trapz(case['scba_probability'], case['x'])):.15g}"
            ),
            (
                f"  w from spectrum={float(case['w_from_spectrum']):.15g}; "
                f"w from scan={float(case['w_from_scan']):.15g}; "
                f"absolute difference={abs(float(case['w_from_spectrum']) - float(case['w_from_scan'])):.3e}"
            ),
        ])
    lines.extend([
        "", "Figure geometry/style", "---------------------",
        "balanced 2x2 layout; equal scan scales and equal spectral scales",
        "panel labels occupy the reserved band above each axes",
        "Exact = black solid; SCBA = red dashed; S uses shared SI colors",
        "positive-J scan markers c and d identify the spectra below",
        "no inset and no A-D cut catalog", "",
    ])
    return "\n".join(lines)


def _manifest_rows(
    source_paths: Sequence[tuple[str, Path, str]],
    generated_paths: Sequence[tuple[str, Path, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for kind, path, role in (*source_paths, *generated_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = sm.project_relative(path, PROJECT_ROOT)
        if relative in seen:
            seen[relative]["role"] = str(seen[relative]["role"]) + "; " + role
            continue
        row = {
            "kind": kind, "path": relative, "sha256": sm.sha256(path),
            "size_bytes": path.stat().st_size, "role": role,
        }
        seen[relative] = row
        rows.append(row)
    rows.sort(key=lambda row: (str(row["kind"]), str(row["path"])))
    return rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only", action="store_true",
        help="refresh processed arrays and validation without rewriting the PDF",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    (
        production_mesh_rows, baseline_rows, _extrema_rows, cut_rows,
        _global_checks, validated_sources,
    ) = sm.load_validated_data(PROJECT_ROOT)
    refined_rows = sm.load_revision27_refined_points(baseline_rows)
    mesh_rows = sm.merge_calculated_mesh(production_mesh_rows, refined_rows)

    spectrum_cases = (
        _unit_area_from_archive(cut_rows, mesh_rows),
        _unit_area_from_solver(mesh_rows),
    )
    for case in spectrum_cases:
        _validate_spectrum(case)

    processed_dir = FIGURE_DIR / "processed"
    mesh_path = processed_dir / "error_mesh.csv"
    baseline_path = processed_dir / "local_baselines.csv"
    validation_path = FIGURE_DIR / "validation.txt"
    manifest_path = FIGURE_DIR / "manifest.csv"
    output_path = FIGURE_DIR / "FigS6.pdf"
    sm.write_csv(mesh_path, mesh_rows, sm.MESH_COLUMNS)
    sm.write_csv(baseline_path, baseline_rows, sm.BASELINE_COLUMNS)
    array_paths, provenance_path = _write_spectrum_products(spectrum_cases)
    sm.write_text(validation_path, _validation_text(mesh_rows, baseline_rows, spectrum_cases))
    if not args.check_only:
        _plot(mesh_rows, baseline_rows, spectrum_cases, output_path)

    source_roles = {
        "source_manifest": "routing for the validated eta/nu0=0.005 profile",
        "manifest": "validated production-profile inventory",
        "summary": "validated 2214-point production scan",
        "baselines": "seven calculated J=0 baselines",
        "cuts": "validated spectral-cut routing including the r=0.800 archive",
        "validation_json": "authoritative production validation record",
        "final_audit": "production profile PASS report",
        "targeted_convergence_csv": "targeted convergence data",
        "targeted_convergence_json": "targeted convergence summary",
        "targeted_convergence_text": "targeted convergence PASS report",
    }
    source_paths = [
        ("input", Path(validated_sources[key]), role)
        for key, role in source_roles.items()
    ]
    source_paths.extend([
        ("input", Path(spectrum_cases[0]["source_path"]), "actual S=0.10, J=+25 meV, r=0.800 spectrum"),
        ("input", REVISION27_DIR / "data/refined_r_points.csv", "34 calculated refined-r scan points"),
        ("input", REVISION27_DIR / "data/summary.json", "revision-27 convergence and solver audit"),
        ("solver", REVISION27_DIR / "audit_dimer_revision27.py", "verified solver path used to regenerate r=1.880"),
        ("solver", REVISION27_DIR.parent / "source/calculate_single_mode_ed.py", "exact-spectrum solver"),
        ("solver", REVISION27_DIR.parent / "source/calculate_single_mode_scba.py", "SCBA spectrum solver"),
        ("solver", REVISION27_DIR.parent / "source/single_mode_common.py", "shared model and grid definitions"),
        ("shared code", Path(sm.__file__).resolve(), "validated data loader and scan merge"),
        ("shared style", PROJECT_ROOT / "Figures/common/si_plot_style.py", "shared SI typography and colors"),
    ])
    generated_paths = [
        ("driver", Path(__file__).resolve(), "active merged Fig. S6 driver"),
        ("documentation", FIGURE_DIR / "README.md", "reproduction and provenance"),
        ("processed", mesh_path, "calculated scan points drawn in panels a and b"),
        ("processed", baseline_path, "J=0 baselines drawn in panels a and b"),
        ("processed", array_paths[0], "unit-area arrays drawn in panel c"),
        ("processed", array_paths[1], "unit-area arrays drawn in panel d"),
        ("processed", provenance_path, "spectral settings, source hashes, areas, and w checks"),
        ("validation", validation_path, "figure-specific PASS report"),
    ]
    if output_path.is_file():
        generated_paths.append(("figure", output_path, "publication vector PDF"))
    sm.write_csv(manifest_path, _manifest_rows(source_paths, generated_paths), MANIFEST_COLUMNS)

    print("Final merged Fig. S6 validation: PASS")
    print(f"scan rows: {len(mesh_rows)}")
    for case in spectrum_cases:
        print(
            "panel (%s): r=%.3f, w=%.15g, |Delta w_scan|=%.3e"
            % (
                case["panel"], float(case["r_abs"]),
                float(case["w_from_spectrum"]),
                abs(float(case["w_from_spectrum"]) - float(case["w_from_scan"])),
            )
        )
    if not args.check_only:
        print(f"wrote {sm.project_relative(output_path, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
