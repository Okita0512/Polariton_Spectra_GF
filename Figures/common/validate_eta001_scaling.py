#!/usr/bin/env python3
"""Recompute the fixed-collective-coupling SCBA linewidth scan.

This validation is deliberately isolated from the legacy manuscript caches.
It imports the production SCBA and linewidth implementations, changes only
the retarded half-width and requested internal grid spacing, and writes every
run beneath ``Figures/common/validation/eta001_scaling``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from scba_g_only import (
    EV_PER_AU,
    interpolate_complex,
    local_green,
    parse_legacy_case,
    solve,
)
from spectral_analysis import TwoPeakMetrics, analyze_two_peaks


ROOT = Path(__file__).resolve().parents[2]
FIG2 = ROOT / "Figures" / "Main" / "Fig2"
OUT = ROOT / "Figures" / "common" / "validation" / "eta001_scaling"
N_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128)
FIT_WINDOWS = {"slow": (1, 2, 3, 4), "fast": (8, 16, 32, 64, 128)}
PRODUCTION_ETA_MEV = 0.010
PRODUCTION_STEP_MEV = 0.010
REFINED_STEP_MEV = 0.005
OUTPUT_MAX_STEP_MEV = 0.005
TOLERANCE_AU = 1.0e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token(value: float) -> str:
    decimals = 4 if 0.0 < abs(value) < 0.001 else 3
    return f"{value:.{decimals}f}".replace(".", "p")


def run_id(bath: str, n_value: int, eta_mev: float, step_mev: float) -> str:
    return (
        f"{bath}_N{n_value:03d}_eta{token(eta_mev)}meV_"
        f"dw{token(step_mev)}meV"
    )


def source_script(bath: str, n_value: int) -> Path:
    return FIG2 / f"N-scaling_{bath}" / str(n_value) / "EGF_sc_fullpol.py"


def fine_output_grid(source_grid: np.ndarray) -> np.ndarray:
    lo = float(source_grid[0] * EV_PER_AU)
    hi = float(source_grid[-1] * EV_PER_AU)
    intervals = max(1, int(np.ceil((hi - lo) * 1.0e3 / OUTPUT_MAX_STEP_MEV)))
    return np.linspace(lo / EV_PER_AU, hi / EV_PER_AU, intervals + 1)


def metric_dict(metrics: TwoPeakMetrics) -> dict[str, object]:
    result: dict[str, object] = {"valley_eV": metrics.valley}
    for label, branch in (("LP", metrics.lower), ("UP", metrics.upper)):
        for key, value in asdict(branch).items():
            result[f"{label}_{key}"] = value
    return result


def analyze_saved(path: Path) -> TwoPeakMetrics:
    data = np.loadtxt(path)
    return analyze_two_peaks(
        data[:, 0] * EV_PER_AU,
        -data[:, 1],
        x_range=(0.8, 3.2),
    )


def analyze_result(case, result) -> TwoPeakMetrics:
    return analyze_two_peaks(
        case.output_grid * EV_PER_AU,
        -np.imag(result.output_green_bright),
        x_range=(0.8, 3.2),
    )


def legacy_grid_reproduction(source_case, result) -> tuple[TwoPeakMetrics, float, float]:
    sigma = interpolate_complex(source_case.output_grid, result.grid, result.sigma)
    _, bright, _ = local_green(source_case.output_grid, sigma, source_case, "full")
    generated = -np.imag(bright)
    metrics = analyze_two_peaks(
        source_case.output_grid * EV_PER_AU,
        generated,
        x_range=(0.8, 3.2),
    )
    saved = np.loadtxt(source_case.script.parent / "Aw_sc_fullpol.txt")
    saved_absorption = -saved[:, 1]
    absolute = float(np.max(np.abs(generated - saved_absorption)))
    relative = absolute / float(np.max(saved_absorption))
    return metrics, absolute, relative


def write_history(path: Path, history: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "iteration",
                "Re_Sigma_at_2eV_au",
                "Im_Sigma_at_2eV_au",
                "max_unmixed_residual_au",
                "mixing",
            ]
        )
        writer.writerows(history)


def run_case(
    bath: str,
    n_value: int,
    eta_mev: float,
    step_mev: float,
    *,
    force: bool = False,
) -> dict[str, object]:
    identifier = run_id(bath, n_value, eta_mev, step_mev)
    directory = OUT / "runs" / identifier
    metadata_path = directory / "metadata.json"
    if metadata_path.exists() and not force:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"CACHE {identifier}: LP={metadata['LP_fwhm'] * 1e3:.9f} meV")
        return metadata

    directory.mkdir(parents=True, exist_ok=True)
    script = source_script(bath, n_value)
    source_case = parse_legacy_case(script)
    case = replace(
        source_case,
        eta=eta_mev / (1.0e3 * EV_PER_AU),
        output_grid=fine_output_grid(source_case.output_grid),
    )
    print(
        f"RUN   {identifier}: {len(case.output_grid)} output points; ",
        flush=True,
        end="",
    )
    result = solve(
        case,
        omega_step_meV=step_mev,
        omega_min_eV=-1.0,
        omega_max_eV=4.0,
        tolerance=TOLERANCE_AU,
        max_iterations=200,
        self_energy_mode="full",
        bath_method="convolution",
    )
    metrics = analyze_result(case, result)
    baseline_metrics = analyze_saved(script.parent / "Aw_sc_fullpol.txt")

    source_eta_mev = source_case.eta * EV_PER_AU * 1.0e3
    legacy_metrics = None
    legacy_abs = np.nan
    legacy_rel = np.nan
    if np.isclose(source_eta_mev, eta_mev, rtol=0.0, atol=1.0e-12):
        legacy_metrics, legacy_abs, legacy_rel = legacy_grid_reproduction(
            source_case, result
        )

    np.savez_compressed(
        directory / "spectrum_and_output_sigma.npz",
        omega_eV=case.output_grid * EV_PER_AU,
        minus_Im_G_bright_au_inv=-np.imag(result.output_green_bright),
        Re_Sigma_au=result.output_sigma.real,
        Im_Sigma_au=result.output_sigma.imag,
    )
    np.savez_compressed(
        directory / "internal_solution.npz",
        omega_eV=result.grid * EV_PER_AU,
        Re_Sigma_au=result.sigma.real,
        Im_Sigma_au=result.sigma.imag,
    )
    write_history(directory / "convergence_history.csv", result.history)

    spacing_mev = float(np.median(np.diff(case.output_grid)) * EV_PER_AU * 1.0e3)
    source_settings = {
        "omega00_eV": source_case.energy_00 * EV_PER_AU,
        "vertical_energy_eV": source_case.vertical_energy * EV_PER_AU,
        "omega_c_eV": source_case.omega_c * EV_PER_AU,
        "lambda_meV": source_case.reorganization * EV_PER_AU * 1.0e3,
        "omega_f_meV": source_case.bath_cutoff * EV_PER_AU * 1.0e3,
        "collective_g_meV": source_case.collective_g * EV_PER_AU * 1.0e3,
        "rabi_splitting_meV": 2.0 * source_case.collective_g * EV_PER_AU * 1.0e3,
        "single_molecule_g_meV": (
            source_case.collective_g * EV_PER_AU * 1.0e3 / np.sqrt(n_value)
        ),
        "cavity_fwhm_meV": 2.0 * source_case.cavity_halfwidth * EV_PER_AU * 1.0e3,
        "temperature_K": source_case.temperature_K,
    }
    metadata: dict[str, object] = {
        "run_id": identifier,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "bath": bath,
        "N": n_value,
        "eta_meV": eta_mev,
        "internal_delta_omega_meV": float(
            np.median(np.diff(result.grid)) * EV_PER_AU * 1.0e3
        ),
        "internal_grid_min_eV": float(result.grid[0] * EV_PER_AU),
        "internal_grid_max_eV": float(result.grid[-1] * EV_PER_AU),
        "internal_n_omega": len(result.grid),
        "output_delta_omega_meV": spacing_mev,
        "output_grid_min_eV": float(case.output_grid[0] * EV_PER_AU),
        "output_grid_max_eV": float(case.output_grid[-1] * EV_PER_AU),
        "output_n_omega": len(case.output_grid),
        "bath_method": result.bath_method,
        "self_energy_mode": result.self_energy_mode,
        "tail_treatment": "asymptotic_Drude",
        "linear_zero_padded_fft": True,
        "source_script": str(script.relative_to(ROOT)),
        "source_script_sha256": sha256(script),
        "iterations": result.iterations,
        "final_unmixed_residual_au": result.final_error,
        "final_unmixed_residual_meV": result.final_error * EV_PER_AU * 1.0e3,
        "final_mixing": result.final_mixing,
        "max_Im_Sigma_au": float(np.max(result.sigma.imag)),
        "min_Im_Sigma_au": float(np.min(result.sigma.imag)),
        "min_minus_Im_G_bright_au_inv": float(
            np.min(-np.imag(result.output_green_bright))
        ),
        "max_minus_Im_G_bright_au_inv": float(
            np.max(-np.imag(result.output_green_bright))
        ),
        "all_finite": bool(
            np.all(np.isfinite(result.sigma))
            and np.all(np.isfinite(result.output_green_bright))
        ),
        "baseline_eta_meV": source_eta_mev,
        "baseline_LP_fwhm_meV": baseline_metrics.lower.fwhm * 1.0e3,
        "baseline_UP_fwhm_meV": baseline_metrics.upper.fwhm * 1.0e3,
        "legacy_grid_reproduction_LP_fwhm_meV": (
            None if legacy_metrics is None else legacy_metrics.lower.fwhm * 1.0e3
        ),
        "legacy_grid_max_abs_spectrum_difference": (
            None if legacy_metrics is None else legacy_abs
        ),
        "legacy_grid_max_relative_spectrum_difference": (
            None if legacy_metrics is None else legacy_rel
        ),
        **source_settings,
        **metric_dict(metrics),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"LP={metrics.lower.fwhm * 1e3:.9f} meV, "
        f"iterations={result.iterations}, residual="
        f"{result.final_error * EV_PER_AU * 1e3:.3e} meV",
        flush=True,
    )
    return metadata


def run_many(cases: Iterable[tuple[str, int, float, float]], force: bool) -> None:
    for bath, n_value, eta_mev, step_mev in cases:
        run_case(bath, n_value, eta_mev, step_mev, force=force)


def fit(rows: list[dict[str, object]], bath: str) -> dict[str, object]:
    selected_n = FIT_WINDOWS[bath]
    selected = sorted(
        (row for row in rows if row["bath"] == bath and int(row["N"]) in selected_n),
        key=lambda row: int(row["N"]),
    )
    if len(selected) != len(selected_n):
        raise RuntimeError(f"Incomplete {bath} fit data")
    x = np.log(np.asarray([int(row["N"]) for row in selected], dtype=float))
    y = np.log(np.asarray([float(row["LP_fwhm"]) * 1.0e3 for row in selected]))
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    residuals = y - predicted
    r_squared = 1.0 - float(np.sum(residuals**2) / np.sum((y - np.mean(y)) ** 2))
    return {
        "bath": bath,
        "N": list(selected_n),
        "slope": float(slope),
        "intercept_ln_meV": float(intercept),
        "R_squared": r_squared,
        "log_residuals": [float(value) for value in residuals],
    }


def load_rows(eta_mev: float, step_mev: float, baths=("slow", "fast")) -> list[dict[str, object]]:
    rows = []
    for bath in baths:
        for n_value in N_VALUES:
            path = OUT / "runs" / run_id(bath, n_value, eta_mev, step_mev) / "metadata.json"
            if path.exists():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def old_fit(bath: str) -> dict[str, object]:
    data = np.loadtxt(FIG2 / f"N-scaling_{bath}" / "width-SCBA.txt")
    included = np.isin(data[:, 0].astype(int), FIT_WINDOWS[bath])
    x = np.log(data[included, 0])
    y = np.log(data[included, 1] * 1.0e3)
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    r_squared = 1.0 - float(np.sum(residuals**2) / np.sum((y - np.mean(y)) ** 2))
    return {
        "bath": bath,
        "slope": float(slope),
        "intercept_ln_meV": float(intercept),
        "R_squared": r_squared,
        "log_residuals": [float(value) for value in residuals],
    }


def write_summary_tables() -> None:
    production = load_rows(PRODUCTION_ETA_MEV, PRODUCTION_STEP_MEV)
    if len(production) != 2 * len(N_VALUES):
        raise RuntimeError(
            f"Production scan incomplete: found {len(production)} of {2 * len(N_VALUES)} cases"
        )
    production.sort(key=lambda row: (str(row["bath"]), int(row["N"])))
    OUT.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "bath",
        "N",
        "eta_meV",
        "internal_delta_omega_meV",
        "LP_fwhm_meV",
        "UP_fwhm_meV",
        "LP_peak_eV",
        "UP_peak_eV",
        "LP_reliable",
        "UP_reliable",
        "LP_weak",
        "UP_weak",
        "LP_prominence",
        "UP_prominence",
        "iterations",
        "final_unmixed_residual_au",
        "final_unmixed_residual_meV",
        "max_Im_Sigma_au",
        "min_minus_Im_G_bright_au_inv",
        "run_id",
        "source_script",
    ]
    with (OUT / "eta001_scaling_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in production:
            writer.writerow(
                {
                    "bath": row["bath"],
                    "N": row["N"],
                    "eta_meV": row["eta_meV"],
                    "internal_delta_omega_meV": row["internal_delta_omega_meV"],
                    "LP_fwhm_meV": float(row["LP_fwhm"]) * 1.0e3,
                    "UP_fwhm_meV": float(row["UP_fwhm"]) * 1.0e3,
                    "LP_peak_eV": row["LP_center"],
                    "UP_peak_eV": row["UP_center"],
                    "LP_reliable": row["LP_reliable"],
                    "UP_reliable": row["UP_reliable"],
                    "LP_weak": row["LP_weak"],
                    "UP_weak": row["UP_weak"],
                    "LP_prominence": row["LP_prominence"],
                    "UP_prominence": row["UP_prominence"],
                    "iterations": row["iterations"],
                    "final_unmixed_residual_au": row["final_unmixed_residual_au"],
                    "final_unmixed_residual_meV": row["final_unmixed_residual_meV"],
                    "max_Im_Sigma_au": row["max_Im_Sigma_au"],
                    "min_minus_Im_G_bright_au_inv": row[
                        "min_minus_Im_G_bright_au_inv"
                    ],
                    "run_id": row["run_id"],
                    "source_script": row["source_script"],
                }
            )

    baseline = {
        bath: {
            int(row[0]): float(row[1] * 1.0e3)
            for row in np.loadtxt(FIG2 / f"N-scaling_{bath}" / "width-SCBA.txt")
        }
        for bath in ("slow", "fast")
    }
    comparison_fields = [
        "bath",
        "N",
        "old_eta_meV",
        "old_LP_fwhm_meV",
        "new_eta_meV",
        "new_LP_fwhm_meV",
        "absolute_change_meV",
        "percent_change",
    ]
    with (OUT / "baseline_vs_eta001.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        for row in production:
            bath = str(row["bath"])
            n_value = int(row["N"])
            old = baseline[bath][n_value]
            new = float(row["LP_fwhm"]) * 1.0e3
            writer.writerow(
                {
                    "bath": bath,
                    "N": n_value,
                    "old_eta_meV": row["baseline_eta_meV"],
                    "old_LP_fwhm_meV": old,
                    "new_eta_meV": row["eta_meV"],
                    "new_LP_fwhm_meV": new,
                    "absolute_change_meV": new - old,
                    "percent_change": 100.0 * (new - old) / old,
                }
            )

    fit_payload = {
        "old": {bath: old_fit(bath) for bath in ("slow", "fast")},
        "eta0p010_dw0p010": {
            bath: fit(production, bath) for bath in ("slow", "fast")
        },
    }
    refined = load_rows(PRODUCTION_ETA_MEV, REFINED_STEP_MEV)
    for bath in ("slow", "fast"):
        if all(
            any(str(row["bath"]) == bath and int(row["N"]) == n for row in refined)
            for n in FIT_WINDOWS[bath]
        ):
            fit_payload.setdefault("eta0p010_dw0p005", {})[bath] = fit(refined, bath)
    eta005_rows = load_rows(0.005, REFINED_STEP_MEV, baths=("fast",))
    if all(
        any(int(row["N"]) == n for row in eta005_rows)
        for n in FIT_WINDOWS["fast"]
    ):
        fit_payload["eta0p005_dw0p005"] = {
            "fast": fit(eta005_rows, "fast")
        }
    (OUT / "fit_results.json").write_text(
        json.dumps(fit_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    production_by_case = {
        (str(row["bath"]), int(row["N"])): row for row in production
    }
    sensitivity_rows = []
    for metadata_path in sorted((OUT / "runs").glob("*/metadata.json")):
        row = json.loads(metadata_path.read_text(encoding="utf-8"))
        reference = production_by_case.get((str(row["bath"]), int(row["N"])))
        if reference is None or row["run_id"] == reference["run_id"]:
            continue
        reference_width = float(reference["LP_fwhm"]) * 1.0e3
        width = float(row["LP_fwhm"]) * 1.0e3
        sensitivity_rows.append(
            {
                "bath": row["bath"],
                "N": row["N"],
                "reference_eta_meV": reference["eta_meV"],
                "reference_delta_omega_meV": reference[
                    "internal_delta_omega_meV"
                ],
                "reference_LP_fwhm_meV": reference_width,
                "test_eta_meV": row["eta_meV"],
                "test_delta_omega_meV": row["internal_delta_omega_meV"],
                "test_LP_fwhm_meV": width,
                "absolute_change_meV": width - reference_width,
                "percent_change": 100.0 * (width - reference_width) / reference_width,
                "run_id": row["run_id"],
            }
        )
    sensitivity_fields = [
        "bath",
        "N",
        "reference_eta_meV",
        "reference_delta_omega_meV",
        "reference_LP_fwhm_meV",
        "test_eta_meV",
        "test_delta_omega_meV",
        "test_LP_fwhm_meV",
        "absolute_change_meV",
        "percent_change",
        "run_id",
    ]
    with (OUT / "numerical_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sensitivity_fields)
        writer.writeheader()
        writer.writerows(sensitivity_rows)


def write_diagnostic_plot() -> None:
    production = load_rows(PRODUCTION_ETA_MEV, PRODUCTION_STEP_MEV)
    if len(production) != 2 * len(N_VALUES):
        raise RuntimeError("Production scan must be complete before plotting")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(4.0, 3.1), constrained_layout=True)
    for bath, marker, color in (
        ("slow", "o", "tab:blue"),
        ("fast", "s", "tab:red"),
    ):
        rows = sorted(
            (row for row in production if row["bath"] == bath),
            key=lambda row: int(row["N"]),
        )
        x = np.log10([int(row["N"]) for row in rows])
        y = np.log10([float(row["LP_fwhm"]) * 1.0e3 for row in rows])
        result = fit(production, bath)
        fit_x = np.log10(np.asarray(result["N"], dtype=float))
        fit_y = result["slope"] * np.log(10.0) * fit_x + result["intercept_ln_meV"]
        fit_y /= np.log(10.0)
        ax.plot(x, y, marker + "-", ms=3.0, lw=0.8, color=color, label=f"{bath}, eta=0.01 meV")
        ax.plot(fit_x, fit_y, "--", lw=0.8, color=color, label=f"{bath} fit, k={result['slope']:.3f}")
    for eta_mev, step_mev, marker, label in (
        (0.005, REFINED_STEP_MEV, "D", r"fast $N=128$, $\eta=0.005$ meV"),
        (0.001, 0.001, "*", r"fast $N=128$, $\eta=0.001$ meV"),
    ):
        path = OUT / "runs" / run_id("fast", 128, eta_mev, step_mev) / "metadata.json"
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            ax.plot(
                np.log10(128.0),
                np.log10(float(row["LP_fwhm"]) * 1.0e3),
                marker=marker,
                ms=5.0,
                mfc="white",
                mec="darkred",
                mew=0.8,
                linestyle="none",
                label=label,
            )
    ax.set_xlabel(r"$\log_{10}N$")
    ax.set_ylabel(r"$\log_{10}(\Gamma_{\rm LP}/{\rm meV})$")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(OUT / "eta001_scaling_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(OUT / "eta001_scaling_diagnostic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_eta_sensitivity_plot() -> None:
    rows = []
    for eta_mev, step_mev in (
        (0.010, PRODUCTION_STEP_MEV),
        (0.005, REFINED_STEP_MEV),
        (0.001, 0.001),
    ):
        path = OUT / "runs" / run_id("fast", 128, eta_mev, step_mev) / "metadata.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    if len(rows) < 2:
        return
    rows.sort(key=lambda row: float(row["eta_meV"]))
    eta = np.asarray([float(row["eta_meV"]) for row in rows])
    widths = np.asarray([float(row["LP_fwhm"]) * 1.0e3 for row in rows])
    fig, ax = plt.subplots(figsize=(3.35, 2.55), constrained_layout=True)
    ax.plot(eta, widths, "o-", color="tab:red", lw=0.9, ms=4.0)
    for x_value, y_value in zip(eta, widths):
        ax.annotate(
            f"{y_value:.6f}",
            (x_value, y_value),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"Retarded half-width $\eta$ (meV)")
    ax.set_ylabel(r"Fast-bath $N=128$ $\Gamma_{\rm LP}$ (meV)")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.savefig(OUT / "eta_sensitivity_fast_N128.pdf", bbox_inches="tight")
    fig.savefig(OUT / "eta_sensitivity_fast_N128.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "stage",
            "production",
            "refine",
            "eta005",
            "eta0001",
            "eta0001_grid",
            "eta00005",
            "summarize",
            "plot",
            "all",
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    stage = [
        ("fast", 8, PRODUCTION_ETA_MEV, PRODUCTION_STEP_MEV),
        ("fast", 16, PRODUCTION_ETA_MEV, PRODUCTION_STEP_MEV),
        ("fast", 128, PRODUCTION_ETA_MEV, PRODUCTION_STEP_MEV),
        ("slow", 4, PRODUCTION_ETA_MEV, PRODUCTION_STEP_MEV),
    ]
    production = [
        (bath, n_value, PRODUCTION_ETA_MEV, PRODUCTION_STEP_MEV)
        for bath in ("slow", "fast")
        for n_value in N_VALUES
    ]
    refinement = [
        ("slow", n_value, PRODUCTION_ETA_MEV, REFINED_STEP_MEV)
        for n_value in (1, 2, 3, 4, 128)
    ] + [
        ("fast", n_value, PRODUCTION_ETA_MEV, REFINED_STEP_MEV)
        for n_value in FIT_WINDOWS["fast"]
    ]
    eta005 = [
        ("slow", n_value, 0.005, REFINED_STEP_MEV)
        for n_value in (1, 2, 3, 4, 128)
    ] + [
        ("fast", n_value, 0.005, REFINED_STEP_MEV)
        for n_value in FIT_WINDOWS["fast"]
    ]
    eta0001 = [
        (bath, 128, 0.001, 0.001) for bath in ("slow", "fast")
    ]
    eta0001_grid = [
        (bath, 128, 0.001, 0.0005) for bath in ("fast", "slow")
    ]
    eta00005 = [
        (bath, 128, 0.0005, 0.0005) for bath in ("fast", "slow")
    ]

    if args.command in ("stage", "all"):
        run_many(stage, args.force)
    if args.command in ("production", "all"):
        run_many(production, args.force)
    if args.command in ("refine", "all"):
        run_many(refinement, args.force)
    if args.command in ("eta005", "all"):
        run_many(eta005, args.force)
    if args.command in ("eta0001", "all"):
        run_many(eta0001, args.force)
    if args.command in ("eta0001_grid", "all"):
        run_many(eta0001_grid, args.force)
    if args.command in ("eta00005", "all"):
        run_many(eta00005, args.force)
    if args.command in ("summarize", "all"):
        write_summary_tables()
    if args.command in ("plot", "all"):
        write_diagnostic_plot()
        write_eta_sensitivity_plot()


if __name__ == "__main__":
    main()
