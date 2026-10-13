#!/usr/bin/env python3
"""Audit and tabulate the command30 collective-linewidth policy.

The script consumes only saved SCBA solutions made by
``validate_eta001_scaling.py``.  It reproduces the manuscript regression,
checks N=128 output-grid sensitivity and spectral weight, and writes the
machine-readable comparison products used for the command30 decision.
"""

from __future__ import annotations

from dataclasses import replace
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scba_g_only import EV_PER_AU, interpolate_complex, local_green, parse_legacy_case
from spectral_analysis import analyze_two_peaks
from validate_eta001_scaling import FIG2, FIT_WINDOWS, N_VALUES, OUT, fit, run_id


GRID_TOLERANCE_MEV = 1.0e-8 * EV_PER_AU * 1.0e3


def load(bath: str, n_value: int, eta: float, step: float) -> dict[str, object]:
    path = OUT / "runs" / run_id(bath, n_value, eta, step) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def eta_intercept(bath: str) -> tuple[float, float, float]:
    points = [
        load(bath, 128, 0.010, 0.010),
        load(bath, 128, 0.005, 0.005),
        load(bath, 128, 0.001, 0.0005),
        load(bath, 128, 0.0005, 0.0005),
    ]
    eta = np.asarray([float(row["eta_meV"]) for row in points])
    width = np.asarray([float(row["LP_fwhm"]) * 1.0e3 for row in points])
    slope, intercept = np.polyfit(eta, width, 1)
    prediction = slope * eta + intercept
    r_squared = 1.0 - float(
        np.sum((width - prediction) ** 2) / np.sum((width - np.mean(width)) ** 2)
    )
    return float(intercept), float(slope), r_squared


def build_schemes() -> dict[str, dict[str, list[dict[str, object]]]]:
    schemes: dict[str, dict[str, list[dict[str, object]]]] = {}
    schemes["A_uniform_eta0p010"] = {
        bath: [load(bath, n, 0.010, 0.010) for n in N_VALUES]
        for bath in ("slow", "fast")
    }
    schemes["B_mixed_N128_eta0p001"] = {}
    schemes["E_tighter_N128_eta0p0005"] = {}
    schemes["D_N128_linear_eta0_diagnostic"] = {}
    for bath in ("slow", "fast"):
        base = [dict(row) for row in schemes["A_uniform_eta0p010"][bath]]
        mixed = [
            load(bath, 128, 0.001, 0.0005) if int(row["N"]) == 128 else row
            for row in base
        ]
        tighter = [
            load(bath, 128, 0.0005, 0.0005) if int(row["N"]) == 128 else row
            for row in base
        ]
        extrapolated = [dict(row) for row in mixed]
        intercept, slope, trend_r2 = eta_intercept(bath)
        for row in extrapolated:
            if int(row["N"]) == 128:
                row["LP_fwhm"] = intercept / 1.0e3
                row["eta_meV"] = 0.0
                row["run_id"] = "N128_linear_eta0_diagnostic"
                row["eta_trend_slope_meV_per_meV"] = slope
                row["eta_trend_R_squared"] = trend_r2
        schemes["B_mixed_N128_eta0p001"][bath] = mixed
        schemes["E_tighter_N128_eta0p0005"][bath] = tighter
        schemes["D_N128_linear_eta0_diagnostic"][bath] = extrapolated
    schemes["C_uniform_eta0p005_fit_windows"] = {
        bath: [load(bath, n, 0.005, 0.005) for n in FIT_WINDOWS[bath]]
        for bath in ("slow", "fast")
    }
    return schemes


def audit_output_grids() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bath in ("slow", "fast"):
        for eta, step in ((0.001, 0.0005), (0.0005, 0.0005)):
            metadata = load(bath, 128, eta, step)
            directory = OUT / "runs" / str(metadata["run_id"])
            with np.load(directory / "internal_solution.npz") as solution:
                internal_grid = solution["omega_eV"] / EV_PER_AU
                sigma = solution["Re_Sigma_au"] + 1.0j * solution["Im_Sigma_au"]
            source_case = parse_legacy_case(Path(metadata["source_script"]))
            case = replace(source_case, eta=eta / (1.0e3 * EV_PER_AU))
            lo, hi = float(source_case.output_grid[0]), float(source_case.output_grid[-1])
            for output_step_mev in (0.010, 0.005, 0.0025, 0.001):
                intervals = int(np.ceil((hi - lo) * EV_PER_AU * 1.0e3 / output_step_mev))
                grid = np.linspace(lo, hi, intervals + 1)
                output_sigma = interpolate_complex(grid, internal_grid, sigma)
                _, bright, _ = local_green(grid, output_sigma, case, "full")
                spectrum = -np.imag(bright)
                metrics = analyze_two_peaks(grid * EV_PER_AU, spectrum, x_range=(0.8, 3.2))
                rows.append(
                    {
                        "bath": bath,
                        "N": 128,
                        "eta_meV": eta,
                        "internal_delta_omega_meV": step,
                        "output_delta_omega_meV": float(np.median(np.diff(grid)) * EV_PER_AU * 1.0e3),
                        "output_n_omega": len(grid),
                        "LP_fwhm_meV": metrics.lower.fwhm * 1.0e3,
                        "UP_fwhm_meV": metrics.upper.fwhm * 1.0e3,
                        "LP_reliable": metrics.lower.reliable,
                        "UP_reliable": metrics.upper.reliable,
                        "min_spectrum_au_inv": float(np.min(spectrum)),
                        "integrated_bright_weight_over_output_range": float(np.trapz(spectrum, grid) / np.pi),
                        "output_min_eV": float(grid[0] * EV_PER_AU),
                        "output_max_eV": float(grid[-1] * EV_PER_AU),
                        "run_id": metadata["run_id"],
                    }
                )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    schemes = build_schemes()
    width_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    for scheme, by_bath in schemes.items():
        for bath, rows in by_bath.items():
            result = fit(rows, bath)
            fit_rows.append(
                {
                    "scheme": scheme,
                    "bath": bath,
                    "fit_N": " ".join(str(n) for n in result["N"]),
                    "k": result["slope"],
                    "intercept_ln_meV": result["intercept_ln_meV"],
                    "R_squared": result["R_squared"],
                }
            )
            for row in sorted(rows, key=lambda item: int(item["N"])):
                width_rows.append(
                    {
                        "scheme": scheme,
                        "bath": bath,
                        "N": row["N"],
                        "in_fit": int(row["N"]) in FIT_WINDOWS[bath],
                        "eta_meV": row["eta_meV"],
                        "internal_delta_omega_meV": row["internal_delta_omega_meV"],
                        "LP_fwhm_meV": float(row["LP_fwhm"]) * 1.0e3,
                        "run_id": row["run_id"],
                    }
                )
    write_csv(OUT / "scaling_eta30_width_comparison.csv", width_rows)
    write_csv(OUT / "scaling_eta30_fit_comparison.csv", fit_rows)

    final_rows = [
        row for row in width_rows if row["scheme"] == "B_mixed_N128_eta0p001"
    ]
    write_csv(OUT / "final_scaling_data_mixed_eta.csv", final_rows)

    output_rows = audit_output_grids()
    write_csv(OUT / "N128_output_grid_and_weight_audit.csv", output_rows)

    endpoint_rows: list[dict[str, object]] = []
    for bath in ("slow", "fast"):
        coarse = load(bath, 128, 0.001, 0.001)
        refined = load(bath, 128, 0.001, 0.0005)
        tighter = load(bath, 128, 0.0005, 0.0005)
        intercept, eta_slope, eta_r2 = eta_intercept(bath)
        endpoint_rows.append(
            {
                "bath": bath,
                "eta0p001_coarse_LP_fwhm_meV": float(coarse["LP_fwhm"]) * 1.0e3,
                "eta0p001_refined_LP_fwhm_meV": float(refined["LP_fwhm"]) * 1.0e3,
                "grid_refinement_percent_change": 100.0 * (float(refined["LP_fwhm"]) / float(coarse["LP_fwhm"]) - 1.0),
                "eta0p0005_LP_fwhm_meV": float(tighter["LP_fwhm"]) * 1.0e3,
                "eta_tightening_percent_change": 100.0 * (float(tighter["LP_fwhm"]) / float(refined["LP_fwhm"]) - 1.0),
                "eta0_linear_intercept_LP_fwhm_meV": intercept,
                "eta_trend_slope_meV_per_meV": eta_slope,
                "eta_trend_R_squared": eta_r2,
                "coarse_residual_meV": coarse["final_unmixed_residual_meV"],
                "refined_residual_meV": refined["final_unmixed_residual_meV"],
                "tighter_residual_meV": tighter["final_unmixed_residual_meV"],
                "residual_tolerance_meV": GRID_TOLERANCE_MEV,
                "refined_max_Im_Sigma_au": refined["max_Im_Sigma_au"],
                "refined_min_spectrum_au_inv": refined["min_minus_Im_G_bright_au_inv"],
                "refined_LP_reliable": refined["LP_reliable"],
                "refined_UP_reliable": refined["UP_reliable"],
                "refined_all_finite": refined["all_finite"],
            }
        )
    write_csv(OUT / "N128_endpoint_audit.csv", endpoint_rows)

    sensitivity_rows: list[dict[str, object]] = []
    for bath in ("slow", "fast"):
        for n_value in FIT_WINDOWS[bath]:
            first = load(bath, n_value, 0.010, 0.010)
            second = load(bath, n_value, 0.005, 0.005)
            first_width = float(first["LP_fwhm"]) * 1.0e3
            second_width = float(second["LP_fwhm"]) * 1.0e3
            sensitivity_rows.append(
                {
                    "bath": bath,
                    "N": n_value,
                    "eta0p010_LP_fwhm_meV": first_width,
                    "eta0p005_LP_fwhm_meV": second_width,
                    "percent_change_eta0p010_to_0p005": 100.0 * (second_width / first_width - 1.0),
                }
            )
    write_csv(OUT / "fit_point_eta_sensitivity.csv", sensitivity_rows)

    plt.rcParams.update({"font.size": 8.0, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    colors = {"slow": "tab:blue", "fast": "tab:red"}
    markers = {"A_uniform_eta0p010": "o", "B_mixed_N128_eta0p001": "s", "C_uniform_eta0p005_fit_windows": "D"}
    labels = {"A_uniform_eta0p010": r"uniform $\eta=0.01$", "B_mixed_N128_eta0p001": r"mixed, $N=128$: $\eta=0.001$", "C_uniform_eta0p005_fit_windows": r"uniform $\eta=0.005$ (fit window)"}
    for axis, bath in zip(axes, ("slow", "fast")):
        for scheme in markers:
            rows = schemes[scheme][bath]
            n = np.asarray([int(row["N"]) for row in rows])
            width = np.asarray([float(row["LP_fwhm"]) * 1.0e3 for row in rows])
            axis.loglog(n, width, marker=markers[scheme], ms=3.5, lw=0.8, label=labels[scheme])
        axis.set_title(f"{bath} bath")
        axis.set_xlabel(r"$N$")
        axis.tick_params(which="both", direction="in", top=True, right=True)
    axes[0].set_ylabel(r"$\Gamma_{\rm LP}$ (meV)")
    axes[1].legend(frameon=False, fontsize=6.8)
    fig.savefig(OUT / "scaling_eta30_diagnostic.pdf", bbox_inches="tight")
    fig.savefig(OUT / "scaling_eta30_diagnostic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.65), constrained_layout=True)
    for axis, bath in zip(axes, ("slow", "fast")):
        points = [
            load(bath, 128, 0.010, 0.010),
            load(bath, 128, 0.005, 0.005),
            load(bath, 128, 0.001, 0.0005),
            load(bath, 128, 0.0005, 0.0005),
        ]
        eta = np.asarray([float(row["eta_meV"]) for row in points])
        width = np.asarray([float(row["LP_fwhm"]) * 1.0e3 for row in points])
        order = np.argsort(eta)
        axis.plot(eta[order], width[order], "o-", color=colors[bath], lw=0.9, ms=3.8)
        axis.set_xscale("log")
        axis.set_title(f"{bath} bath, $N=128$")
        axis.set_xlabel(r"$\eta$ (meV)")
        axis.tick_params(which="both", direction="in", top=True, right=True)
    axes[0].set_ylabel(r"$\Gamma_{\rm LP}$ (meV)")
    fig.savefig(OUT / "N128_eta_sensitivity30.pdf", bbox_inches="tight")
    fig.savefig(OUT / "N128_eta_sensitivity30.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    payload = {"fits": fit_rows, "N128": endpoint_rows}
    (OUT / "scaling_eta30_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
