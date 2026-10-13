#!/usr/bin/env python3
"""Generate shifted-frequency G-only SCBA data and manuscript figures.

The ``--figures-only`` path is intentionally limited to plotting existing
cached data.  It never invokes an SCBA, HEOM, or other calculation script.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scba_g_only import (
    EV_PER_AU,
    run_legacy_ba_script,
    run_legacy_script,
)
from rebuild_analysis import main as rebuild_analysis_tables
from spectral_analysis import analyze_two_peaks


ROOT = Path(__file__).resolve().parent.parent.parent
VALIDATION_DIR = ROOT / "Figures" / "common" / "validation"
SOLVER_ROOTS = (
    ROOT / "Figures" / "SI" / "FigS12",
    ROOT / "Figures" / "Main" / "Fig2",
    ROOT / "Figures" / "shared" / "detuning" / "cache",
    ROOT / "Figures" / "SI" / "FigS1",
)

# These are the external manuscript figures. Main Fig. 1 and SI Fig. S1 are
# drawn in TeX; the SCBA algorithm in SI Sec. IV B is also typeset in TeX.
PLOT_COMMANDS: Sequence[Tuple[str, Tuple[str, ...]]] = (
    ("main Fig. 2", (sys.executable, "Figures/Main/Fig2/plot_Fig2.py")),
    ("main Fig. 3", (sys.executable, "Figures/Main/Fig3/plot_Fig3.py")),
    ("main Fig. 4", (sys.executable, "Figures/Main/Fig4/plot_Fig4.py")),
    ("SI Fig. S2", (sys.executable, "Figures/SI/FigS2/plot_FigS2.py")),
    ("SI Fig. S3", (sys.executable, "Figures/SI/FigS3/plot_FigS3.py")),
    ("SI Fig. S4", (sys.executable, "Figures/SI/FigS4/plot_FigS4.py")),
    ("SI Fig. S5", (sys.executable, "Figures/SI/FigS5/plot_FigS5.py")),
    ("SI Fig. S6", (sys.executable, "Figures/SI/FigS6/plot_FigS6.py")),
    ("SI Fig. S7", (sys.executable, "Figures/SI/FigS7/plot_FigS7.py")),
    ("SI Fig. S8", ("bash", "Figures/SI/FigS8/build_FigS8.sh")),
    ("SI Fig. S9", (sys.executable, "Figures/SI/FigS9/plot_FigS9.py")),
    ("SI Fig. S10", (sys.executable, "Figures/SI/FigS10/plot_FigS10.py")),
    ("SI Fig. S11", (sys.executable, "Figures/SI/FigS11/plot_FigS11.py")),
    ("SI Fig. S12", (sys.executable, "Figures/SI/FigS12/plot_FigS12.py")),
)


def discover_scripts() -> List[Path]:
    scripts: List[Path] = []
    for solver_root in SOLVER_ROOTS:
        for script in solver_root.rglob("EGF_sc_fullpol.py"):
            parts = {part.casefold() for part in script.parts}
            if parts & {"trash", "unused", "archive"}:
                continue
            scripts.append(script.resolve())
    return sorted(scripts)


def two_peak_metrics(path: Path) -> Tuple[float, float, float, float]:
    data = np.loadtxt(str(path))
    x = data[:, 0] * EV_PER_AU
    y = -data[:, 1]
    try:
        metrics = analyze_two_peaks(x, y)
    except ValueError:
        return np.nan, np.nan, np.nan, np.nan
    return (
        metrics.lower.fwhm,
        metrics.upper.fwhm,
        metrics.lower.center,
        metrics.upper.center,
    )


def write_metrics(
    entries: Iterable[Tuple[float, Path]],
    width_path: Path,
    peak_path: Path,
) -> None:
    widths = []
    peaks = []
    for coordinate, spectrum in entries:
        lp_width, up_width, lp_peak, up_peak = two_peak_metrics(spectrum)
        widths.append((coordinate, lp_width, up_width))
        peaks.append((coordinate, lp_peak, up_peak))
    np.savetxt(
        str(width_path),
        np.asarray(widths),
        header="coordinate LP_FWHM_eV UP_FWHM_eV; G-only SCBA",
    )
    np.savetxt(
        str(peak_path),
        np.asarray(peaks),
        header="coordinate LP_peak_eV UP_peak_eV; G-only SCBA",
    )


def generate_derived_tables() -> None:
    rebuild_analysis_tables()


def generate_figures() -> None:
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    for label, command in PLOT_COMMANDS:
        print("plot: {} ({})".format(label, command[-1]))
        subprocess.run(
            command,
            cwd=str(ROOT),
            env=environment,
            check=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scripts",
        nargs="*",
        type=Path,
        help="Optional EGF_sc_fullpol.py paths; default is all retained cases.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument("--n-bath", type=int, default=1000)
    parser.add_argument("--n-omega", type=int)
    parser.add_argument("--omega-step-mev", type=float)
    parser.add_argument(
        "--bath-method",
        choices=("convolution", "tangent"),
        default="convolution",
    )
    parser.add_argument("--skip-derived", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help=(
            "Rebuild every external manuscript figure from cached inputs "
            "without running calculation scripts."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.figures_only:
        generate_figures()
        return

    scripts = (
        [path.resolve() for path in args.scripts]
        if args.scripts
        else discover_scripts()
    )
    print(
        "Generating {} cases with {} workers, bath_method={}, "
        "n_bath={}, n_omega={}, omega_step_meV={}".format(
            len(scripts),
            args.workers,
            args.bath_method,
            args.n_bath,
            args.n_omega,
            args.omega_step_mev,
        )
    )
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_legacy_script,
                script,
                args.n_bath,
                args.n_omega,
                None,
                args.bath_method,
                args.omega_step_mev,
            ): script
            for script in scripts
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            summary = future.result()
            summaries.append(summary)
            print(
                "[{}/{}] {}".format(
                    completed,
                    len(scripts),
                    Path(summary["script"]).parent.relative_to(ROOT),
                )
            )

    ba_script = (
        ROOT
        / "Figures"
        / "SI"
        / "FigS12"
        / "N=1"
        / "EGF_single_shot_ba_fullpol.py"
    )
    if not args.scripts or ba_script.resolve() in scripts:
        summaries.append(
            run_legacy_ba_script(
                ba_script,
                n_bath=args.n_bath,
                n_omega=args.n_omega,
                bath_method=args.bath_method,
                omega_step_meV=args.omega_step_mev,
            )
        )

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = VALIDATION_DIR / "scba_generation_summary.csv"
    fields = (
        "script",
        "n_molecules",
        "omega_c_eV",
        "iterations",
        "residual_meV",
        "mixing",
        "n_bath",
        "n_omega",
        "omega_step_meV",
        "bath_method",
        "self_energy_mode",
    )
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(summaries, key=lambda item: item["script"]))

    if not args.skip_derived:
        generate_derived_tables()
    if not args.skip_figures:
        generate_figures()


if __name__ == "__main__":
    main()
