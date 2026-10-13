#!/usr/bin/env python3
"""Rebuild all manuscript linewidth and peak tables with one protocol."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from spectral_analysis import TwoPeakMetrics, analyze_two_peaks


ROOT = Path(__file__).resolve().parent.parent.parent
VALIDATION = ROOT / "Figures" / "common" / "validation"
MAIN_FIG2_ROOT = ROOT / "Figures" / "Main" / "Fig2"
DETUNING_CACHE_ROOT = ROOT / "Figures" / "shared" / "detuning" / "cache"
SI_FIGS1_ROOT = ROOT / "Figures" / "SI" / "FigS1"
EV_PER_AU = 27.211397
N_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128)
DETUNINGS = (
    "1.0660571",
    "1.2494866",
    "1.4375644",
    "1.4953152",
    "1.6368",
    "1.7",
    "1.8",
    "1.85",
    "1.9",
    "1.95",
    "2.0",
    "2.05",
    "2.1",
    "2.15",
    "2.2",
    "2.25",
    "2.3",
    "2.3632",
    "2.5046848",
    "2.5624356",
    "2.7505134",
    "2.9339429",
)
REFERENCE_DETUNINGS = (
    "1.7",
    "1.8",
    "1.85",
    "1.9",
    "1.95",
    "2.0",
    "2.05",
    "2.1",
    "2.15",
    "2.2",
    "2.3",
)


def _scba_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    return data[:, 0] * EV_PER_AU, -data[:, 1]


def _heom_spectrum(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.loadtxt(directory / "resp1st.w1") * EV_PER_AU,
        np.loadtxt(directory / "resp1st_im.w"),
    )


def _mf_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        header = handle.readline()
    if "MinusImChi_eV^-1" not in header:
        raise ValueError(
            "Stale MF-HEOM observable (expected -Im[P/E]); run "
            f"Figures/common/process_mf_heom.py for {path.parent}"
        )
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def _analyze(
    x: np.ndarray,
    y: np.ndarray,
    x_range: tuple[float, float] = (0.8, 3.2),
) -> TwoPeakMetrics:
    return analyze_two_peaks(x, y, x_range=x_range)


def _save_table(
    coordinates: Sequence[float],
    metrics: Sequence[TwoPeakMetrics],
    width_path: Path,
    peak_path: Path,
    coordinate_name: str,
    method: str,
) -> None:
    widths = np.asarray(
        [
            (coordinate, item.lower.fwhm, item.upper.fwhm)
            for coordinate, item in zip(coordinates, metrics)
        ]
    )
    peaks = np.asarray(
        [
            (coordinate, item.lower.center, item.upper.center)
            for coordinate, item in zip(coordinates, metrics)
        ]
    )
    np.savetxt(
        width_path,
        widths,
        header=(
            f"{coordinate_name} LP_FWHM_eV UP_FWHM_eV; {method}; "
            "direct local-baseline FWHM"
        ),
    )
    np.savetxt(
        peak_path,
        peaks,
        header=f"{coordinate_name} LP_peak_eV UP_peak_eV; {method}",
    )


def _status_rows(
    figure: str,
    method: str,
    coordinates: Iterable[float],
    metrics: Iterable[TwoPeakMetrics],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for coordinate, item in zip(coordinates, metrics):
        for branch, branch_metrics in (
            ("LP", item.lower),
            ("UP", item.upper),
        ):
            rows.append(
                {
                    "figure": figure,
                    "method": method,
                    "coordinate": coordinate,
                    "branch": branch,
                    "center_eV": branch_metrics.center,
                    "FWHM_eV": branch_metrics.fwhm,
                    "left_half_width_eV": branch_metrics.left_half_width,
                    "right_half_width_eV": branch_metrics.right_half_width,
                    "prominence": branch_metrics.prominence,
                    "grid_uncertainty_eV": branch_metrics.uncertainty,
                    "reliable": int(branch_metrics.reliable),
                    "weak": int(branch_metrics.weak),
                    "manual": 0,
                }
            )
    return rows


def rebuild_main_fig2(status: list[dict[str, object]]) -> None:
    for speed in ("slow", "fast"):
        directory = MAIN_FIG2_ROOT / f"N-scaling_{speed}"
        scba = [
            _analyze(*_scba_spectrum(directory / str(n) / "Aw_sc_fullpol.txt"))
            for n in N_VALUES
        ]
        _save_table(
            N_VALUES,
            scba,
            directory / "width-SCBA.txt",
            directory / "peaks-SCBA.txt",
            "N",
            "full SCBA",
        )
        status.extend(
            _status_rows(f"MainFig2_{speed}", "SCBA", N_VALUES, scba)
        )

        heom_n = tuple(
            n
            for n in N_VALUES
            if (directory / str(n) / "resp1st.w1").exists()
        )
        heom = [
            _analyze(*_heom_spectrum(directory / str(n)))
            for n in heom_n
        ]
        _save_table(
            heom_n,
            heom,
            directory / "width-HEOM.txt",
            directory / "peaks-HEOM.txt",
            "N",
            "HEOM",
        )
        status.extend(
            _status_rows(f"MainFig2_{speed}", "HEOM", heom_n, heom)
        )


def rebuild_detuning_tables(status: list[dict[str, object]]) -> None:
    root = DETUNING_CACHE_ROOT
    for n in (1, 2, 4, 8, 16, 32, 64, 10000):
        case_root = root / f"N={n}_Gammac=44.15mev"
        labels = tuple(
            label
            for label in DETUNINGS
            if (case_root / label / "Aw_sc_fullpol.txt").exists()
        )
        cavity = tuple(float(label) + 0.03 for label in labels)
        scba = [
            _analyze(
                *_scba_spectrum(case_root / label / "Aw_sc_fullpol.txt"),
                x_range=(0.9, 3.15),
            )
            for label in labels
        ]
        _save_table(
            cavity,
            scba,
            root / f"width_N={n}_Gammac=44.15mev-SCBA.txt",
            root / f"peaks_N={n}_Gammac=44.15mev-SCBA.txt",
            "omega_c_eV",
            "full SCBA",
        )
        scba_status = _status_rows(
            f"Detuning_N{n}", "SCBA", cavity, scba
        )
        # The near-pure 0.97/0.98 cases are included for only their
        # exciton-rich branch. Their complementary 0.03/0.02 photon peaks
        # are outside the plotted sampling and are not validation targets.
        scba_status = [
            row
            for row in scba_status
            if not (
                (
                    np.isclose(
                        float(row["coordinate"]),
                        (1.0960571, 1.2794866),
                        rtol=0.0,
                        atol=1.0e-7,
                    ).any()
                    and row["branch"] == "LP"
                )
                or (
                    np.isclose(
                        float(row["coordinate"]),
                        (2.7805134, 2.9639429),
                        rtol=0.0,
                        atol=1.0e-7,
                    ).any()
                    and row["branch"] == "UP"
                )
            )
        ]
        status.extend(scba_status)

        if n <= 8:
            heom_labels = tuple(
                label
                for label in REFERENCE_DETUNINGS
                if (case_root / label / "resp1st.w1").exists()
            )
            heom_cases = [
                (
                    float(label) + 0.03,
                    _analyze(
                        *_heom_spectrum(case_root / label),
                        x_range=(1.4, 2.6),
                    ),
                )
                for label in heom_labels
            ]

            # Additional small-N HEOM references are stored separately from
            # the mixed SCBA/legacy-reference directories and use their
            # physical cavity frequency in the directory name.
            organized_heom_root = root / "HEOM" / f"N={n}"
            for directory in sorted(
                organized_heom_root.glob("wc=*"),
                key=lambda path: float(path.name.split("=", 1)[1]),
            ):
                spectrum = directory / "resp1st.w1"
                if not spectrum.exists():
                    continue
                coordinate = float(directory.name.split("=", 1)[1])
                if any(
                    np.isclose(coordinate, item[0])
                    for item in heom_cases
                ):
                    raise ValueError(
                        "Duplicate detuning HEOM cavity frequency for "
                        f"N={n}: {coordinate}"
                    )
                heom_cases.append(
                    (
                        coordinate,
                        _analyze(
                            *_heom_spectrum(directory),
                            x_range=(0.9, 3.15),
                        ),
                    )
                )

            heom_cases.sort(key=lambda item: item[0])
            heom_cavity = tuple(item[0] for item in heom_cases)
            heom = [item[1] for item in heom_cases]
            _save_table(
                heom_cavity,
                heom,
                root / f"width_N={n}_Gammac=44.15mev-HEOM.txt",
                root / f"peaks_N={n}_Gammac=44.15mev-HEOM.txt",
                "omega_c_eV",
                "HEOM",
            )
            heom_status = _status_rows(
                f"Detuning_N{n}", "HEOM", heom_cavity, heom
            )
            # The new low-cavity-frequency calculations target the
            # exciton-rich UP. Retain their complementary photon-rich LPs
            # in the machine-readable tables, but do not display or count
            # them as figure measurements.
            status.extend(
                row
                for row in heom_status
                if not (
                    np.isclose(float(row["coordinate"]), 1.28)
                    and row["branch"] == "LP"
                )
            )

    mf_root = root / "N=10000_Gammac=44.15mev"
    labels = tuple(
        label
        for label in REFERENCE_DETUNINGS
        if (mf_root / label / "absorption_pulse_2eV.txt").exists()
    )
    mf_cases = [
        (
            float(label) + 0.03,
            _analyze(
                *_mf_spectrum(
                    mf_root / label / "absorption_pulse_2eV.txt"
                ),
                x_range=(1.4, 2.6),
            ),
        )
        for label in labels
    ]

    # New MF-HEOM reference runs are stored separately from the mixed
    # SCBA/legacy-reference directories and use their physical cavity
    # frequency in the directory name.
    organized_mf_root = root / "MF-HEOM"
    for directory in sorted(
        organized_mf_root.glob("wc=*"),
        key=lambda path: float(path.name.split("=", 1)[1]),
    ):
        spectrum = directory / "absorption_pulse_2eV.txt"
        if not spectrum.exists():
            continue
        coordinate = float(directory.name.split("=", 1)[1])
        if any(np.isclose(coordinate, item[0]) for item in mf_cases):
            raise ValueError(
                "Duplicate detuning MF-HEOM cavity frequency: "
                f"{coordinate}"
            )
        mf_cases.append(
            (
                coordinate,
                _analyze(
                    *_mf_spectrum(spectrum),
                    x_range=(0.9, 3.15),
                ),
            )
        )

    mf_cases.sort(key=lambda item: item[0])
    cavity = tuple(item[0] for item in mf_cases)
    mf = [item[1] for item in mf_cases]
    _save_table(
        cavity,
        mf,
        root / "width_N=10000_Gammac=44.15mev-MF-HEOM.txt",
        root / "peaks_N=10000_Gammac=44.15mev-MF-HEOM.txt",
        "omega_c_eV",
        "MF-HEOM",
    )
    mf_status = _status_rows("Detuning_Ninf", "MF-HEOM", cavity, mf)
    # The new low-cavity-frequency calculation targets the exciton-rich UP.
    # Its complementary photon-rich LP is retained in the machine-readable
    # width table but is not displayed or counted as a figure measurement.
    status.extend(
        row
        for row in mf_status
        if not (
            np.isclose(float(row["coordinate"]), 1.28)
            and row["branch"] == "LP"
        )
    )


def rebuild_si_figs1(status: list[dict[str, object]]) -> None:
    root = SI_FIGS1_ROOT
    directories = sorted(
        root.glob("wc=*"),
        key=lambda path: float(path.name.split("=", 1)[1]),
    )
    cavity = tuple(float(path.name.split("=", 1)[1]) for path in directories)
    scba = [
        _analyze(
            *_scba_spectrum(path / "Aw_sc_fullpol.txt"),
            x_range=(0.8, 3.1),
        )
        for path in directories
    ]
    mf = [
        _analyze(
            *_mf_spectrum(path / "absorption_pulse_2eV.txt"),
            x_range=(0.8, 3.1),
        )
        for path in directories
    ]
    _save_table(
        cavity,
        scba,
        root / "width-SCBA.txt",
        root / "peaks-SCBA.txt",
        "omega_c_eV",
        "full SCBA",
    )
    _save_table(
        cavity,
        mf,
        root / "width-MF.txt",
        root / "peaks-MF.txt",
        "omega_c_eV",
        "MF-HEOM",
    )
    status.extend(_status_rows("FigS1", "SCBA", cavity, scba))
    status.extend(_status_rows("FigS1", "MF-HEOM", cavity, mf))


def write_fit_table() -> None:
    """Record the predeclared main-Fig.-2 fit windows and Kubo parameters."""
    temperature_eV = 300.0 / 11604.51812
    reorganization_eV = 0.030
    rows = []
    for speed, cutoff_eV, included in (
        ("slow", 0.00248, {1, 2, 3, 4}),
        ("fast", 0.0248, {8, 16, 32, 64, 128}),
    ):
        data = np.loadtxt(
            MAIN_FIG2_ROOT / f"N-scaling_{speed}" / "width-SCBA.txt"
        )
        n_values = data[:, 0].astype(int)
        widths = data[:, 1]
        local = np.full_like(widths, np.nan)
        local[1:] = np.diff(np.log(widths)) / np.diff(np.log(n_values))
        for n, width, exponent in zip(n_values, widths, local):
            lambda_effective = reorganization_eV / n
            kubo = cutoff_eV / np.sqrt(
                2.0 * lambda_effective * temperature_eV
            )
            rows.append(
                {
                    "bath": speed,
                    "N": n,
                    "K_N": kubo,
                    "LP_FWHM_meV": 1000.0 * width,
                    "included_in_fit": int(n in included),
                    "local_exponent_from_previous_N": exponent,
                }
            )
    VALIDATION.mkdir(parents=True, exist_ok=True)
    # Retain the established artifact name for downstream compatibility even
    # though its source data now belong to main Fig. 2.
    path = VALIDATION / "fig4_fit_windows.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    status: list[dict[str, object]] = []
    rebuild_main_fig2(status)
    rebuild_detuning_tables(status)
    rebuild_si_figs1(status)
    write_fit_table()
    VALIDATION.mkdir(parents=True, exist_ok=True)
    with (VALIDATION / "linewidth_status.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(status[0]))
        writer.writeheader()
        writer.writerows(status)
    unreliable = sum(not int(row["reliable"]) for row in status)
    weak = sum(int(row["weak"]) for row in status)
    print(
        f"Rebuilt {len(status)} branch measurements: "
        f"{unreliable} unreliable, {weak} weak, 0 manual."
    )


if __name__ == "__main__":
    main()
