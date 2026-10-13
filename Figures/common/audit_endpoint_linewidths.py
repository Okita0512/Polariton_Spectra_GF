#!/usr/bin/env python3
"""Audit and archive the three uncoupled-molecule direct-FWHM routes.

The canonical reporting values are obtained only after applying the shared
direct-FWHM routine to the raw HEOM, production FFT-convolution SCBA, and
independent tangent-grid SCBA spectra.  The JSON written here is the sole
source consumed by the detuning figures, Fig. S11, and Fig. S12.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


COMMON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = COMMON_DIR.parent.parent
FIGS11_DIR = PROJECT_ROOT / "Figures" / "SI" / "FigS11"
sys.path.insert(0, str(COMMON_DIR))
sys.path.insert(0, str(FIGS11_DIR))

from spectral_analysis import BranchMetrics, analyze_single_peak  # noqa: E402
import diagnose_scba as figs11  # noqa: E402


OUTPUT = COMMON_DIR / "endpoint_linewidths.json"
EV_PER_AU = 27.211397


def grid_metadata(x: np.ndarray) -> dict[str, Any]:
    ordered = np.sort(np.asarray(x, dtype=float))
    return {
        "points": int(len(ordered)),
        "minimum_eV": float(ordered[0]),
        "maximum_eV": float(ordered[-1]),
        "median_step_meV": float(1.0e3 * np.median(np.diff(ordered))),
    }


def metrics_metadata(metrics: BranchMetrics) -> dict[str, Any]:
    return {
        "peak_eV": metrics.center,
        "fwhm_meV": 1.0e3 * metrics.fwhm,
        "left_halfwidth_meV": 1.0e3 * metrics.left_half_width,
        "right_halfwidth_meV": 1.0e3 * metrics.right_half_width,
        "normalized_baseline": metrics.baseline,
        "normalized_peak_height": metrics.peak_height,
        "prominence": metrics.prominence,
        "reliable": metrics.reliable,
        "fwhm_resolution_meV": 1.0e3 * metrics.uncertainty,
    }


def production_header(path: Path) -> dict[str, Any]:
    first_line = path.open(encoding="utf-8").readline().strip()
    match = re.search(
        r"bath_method=(\S+).*n_omega=(\d+) "
        r"omega_step_meV=([0-9.eE+-]+) "
        r"omega_grid_eV=\[([0-9.eE+-]+),([0-9.eE+-]+)\]",
        first_line,
    )
    if match is None:
        raise ValueError(f"Could not parse numerical header in {path}")
    return {
        "bath_method": match.group(1),
        "points": int(match.group(2)),
        "step_meV": float(match.group(3)),
        "minimum_eV": float(match.group(4)),
        "maximum_eV": float(match.group(5)),
    }


def final_scba_residual(path: Path) -> dict[str, Any]:
    table = np.loadtxt(path)
    return {
        "iterations": int(table[-1, 0]),
        "final_residual_hartree": float(table[-1, 3]),
        "final_residual_eV": float(table[-1, 3] * EV_PER_AU),
        "final_mixing": float(table[-1, 4]),
    }


def build_audit() -> dict[str, Any]:
    production_dir = (
        PROJECT_ROOT / "Figures" / "Main" / "Fig2" / "fast" / "N=1_out"
    )

    heom_energy = np.loadtxt(production_dir / "resp1st.w1") * EV_PER_AU
    heom_spectrum = np.loadtxt(production_dir / "resp1st_im.w")
    heom_support = (1.5, 2.5)
    heom_metrics = analyze_single_peak(
        heom_energy,
        heom_spectrum,
        x_range=heom_support,
    )

    fft_path = production_dir / "Aw_sc_fullpol.txt"
    fft_spectrum = np.loadtxt(fft_path)
    fft_energy = fft_spectrum[:, 0] * EV_PER_AU
    fft_support = (1.5, 2.5)
    fft_metrics = analyze_single_peak(
        fft_energy,
        -fft_spectrum[:, 1],
        x_range=fft_support,
    )

    settings = figs11.Settings()
    tangent_tag = figs11.default_cache_tag(
        "g0",
        1,
        None,
        figs11.TEMPERATURE_K,
        "full",
        settings,
    )
    tangent_path = figs11.cache_file(tangent_tag)
    tangent_cache = np.load(tangent_path, allow_pickle=False)
    tangent_case = figs11.base_case(
        1,
        figs11.VERTICAL_ENERGY_EV,
        collective_g_eV=0.0,
    )
    tangent_energy = tangent_case.output_grid * figs11.EV_PER_AU
    tangent_support = (1.65, 2.35)
    tangent_metrics = analyze_single_peak(
        tangent_energy,
        -np.imag(tangent_cache["output_green_bright"]),
        x_range=tangent_support,
    )

    metrics = (heom_metrics, fft_metrics, tangent_metrics)
    if not all(item.reliable for item in metrics):
        raise RuntimeError("At least one endpoint direct FWHM is unreliable")

    heom_raw = 1.0e3 * heom_metrics.fwhm
    fft_raw = 1.0e3 * fft_metrics.fwhm
    tangent_raw = 1.0e3 * tangent_metrics.fwhm
    canonical_heom = round(heom_raw, 2)
    canonical_scba = round(tangent_raw, 3)

    archived_row = np.genfromtxt(
        FIGS11_DIR / "data" / "isolated_molecule.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    archived_tangent = float(archived_row["fwhm_meV"])
    if not np.isclose(
        archived_tangent,
        tangent_raw,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise RuntimeError(
            "Fig. S13 isolated-molecule CSV disagrees with its raw cache"
        )

    return {
        "schema_version": 1,
        "units": "meV",
        "analysis": {
            "routine": "Figures/common/spectral_analysis.py:analyze_single_peak",
            "definition": "direct full width at half maximum",
            "normalization": "divide the original spectrum by its maximum",
            "baseline": "minimum of the original spectrum on the stated line support",
            "crossings": "linear interpolation on the original unsmoothed grid",
            "fit": None,
        },
        "canonical": {
            "heom_molecule_fwhm_meV": canonical_heom,
            "scba_molecule_fwhm_meV": canonical_scba,
            "eq26_gamma_ex_meV": canonical_heom,
            "reporting_policy": {
                "heom_decimals": 2,
                "scba_precise_decimals": 3,
                "scba_caption_decimals": 2,
            },
        },
        "routes": {
            "heom_uncoupled": {
                "source_energy": (
                    "Figures/Main/Fig2/fast/N=1_out/resp1st.w1"
                ),
                "source_spectrum": (
                    "Figures/Main/Fig2/fast/N=1_out/resp1st_im.w"
                ),
                "line_support_eV": list(heom_support),
                "raw_grid": grid_metadata(heom_energy),
                "numerical_broadening_meV": None,
                "response_settings": {
                    "time_step_au": 0.41341,
                    "time_step_fs": 0.01,
                    "time_points": 4000,
                    "postprocessing_broadening": "none",
                },
                **metrics_metadata(heom_metrics),
            },
            "scba_production_fft": {
                "source_spectrum": (
                    "Figures/Main/Fig2/fast/N=1_out/Aw_sc_fullpol.txt"
                ),
                "line_support_eV": list(fft_support),
                "raw_output_grid": grid_metadata(fft_energy),
                "internal_grid": production_header(fft_path),
                "numerical_broadening_meV": 0.1,
                "self_energy_interpolation": "linear",
                "tolerance_hartree": 1.0e-8,
                "tolerance_eV": 1.0e-8 * EV_PER_AU,
                **final_scba_residual(
                    production_dir / "scba_convergence.txt"
                ),
                **metrics_metadata(fft_metrics),
            },
            "scba_independent_tangent": {
                "source_cache": str(tangent_path.relative_to(PROJECT_ROOT)),
                "source_table": (
                    "Figures/SI/FigS11/data/isolated_molecule.csv"
                ),
                "line_support_eV": list(tangent_support),
                "raw_output_grid": grid_metadata(tangent_energy),
                "internal_grid": {
                    "bath_method": "tangent",
                    "bath_points": settings.n_bath,
                    "points": settings.n_omega,
                    "minimum_eV": settings.omega_min_eV,
                    "maximum_eV": settings.omega_max_eV,
                    "step_meV": (
                        1.0e3
                        * (settings.omega_max_eV - settings.omega_min_eV)
                        / (settings.n_omega - 1)
                    ),
                },
                "numerical_broadening_meV": 1.0e3 * figs11.ETA_EV,
                "self_energy_interpolation": "linear",
                "tolerance_hartree": settings.tolerance,
                "tolerance_eV": settings.tolerance * figs11.EV_PER_AU,
                "max_iterations": settings.max_iterations,
                "iterations": int(tangent_cache["iterations"]),
                "final_residual_hartree": float(
                    tangent_cache["final_error"]
                ),
                "final_residual_eV": float(
                    tangent_cache["final_error"] * figs11.EV_PER_AU
                ),
                "final_mixing": float(tangent_cache["final_mixing"]),
                **metrics_metadata(tangent_metrics),
            },
        },
        "cross_route": {
            "production_fft_minus_independent_tangent_meV": (
                fft_raw - tangent_raw
            ),
            "absolute_difference_meV": abs(fft_raw - tangent_raw),
            "relative_difference_percent": (
                100.0 * abs(fft_raw - tangent_raw) / tangent_raw
            ),
            "canonical_scba_route": "scba_independent_tangent",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the raw-spectrum audit with the archived JSON",
    )
    args = parser.parse_args()
    audit = build_audit()
    serialized = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != serialized:
            raise SystemExit(
                f"{OUTPUT} is stale; run {Path(__file__).relative_to(PROJECT_ROOT)}"
            )
        print(f"PASS: {OUTPUT.relative_to(PROJECT_ROOT)} matches all raw spectra")
    else:
        OUTPUT.write_text(serialized, encoding="utf-8")
        print(f"Wrote {OUTPUT.relative_to(PROJECT_ROOT)}")
    print(
        "Direct FWHM (meV): "
        f"HEOM={audit['routes']['heom_uncoupled']['fwhm_meV']:.9f}, "
        "SCBA FFT="
        f"{audit['routes']['scba_production_fft']['fwhm_meV']:.9f}, "
        "SCBA tangent="
        f"{audit['routes']['scba_independent_tangent']['fwhm_meV']:.9f}"
    )
    print(
        "SCBA cross-route difference (FFT - tangent): "
        f"{audit['cross_route']['production_fft_minus_independent_tangent_meV']:.9f} meV"
    )


if __name__ == "__main__":
    main()
