#!/usr/bin/env python3
"""Reconstruct positive-frequency MF-HEOM susceptibility spectra."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


EV_PER_AU = 27.211397
FS_TO_AU = 41.341374575751


def _pulse_parameters(script: Path) -> Dict[str, float]:
    wanted = {
        "conv",
        "fs_to_au",
        "pulse_ampl",
        "pulse_freq",
        "pulse_sigm",
        "pulse_varp",
        "pulse_t0",
    }
    values: Dict[str, object] = {"np": np}
    tree = ast.parse(script.read_text(), filename=str(script))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in wanted:
            continue
        expression = ast.Expression(body=node.value)
        ast.fix_missing_locations(expression)
        values[target.id] = eval(
            compile(expression, str(script), "eval"),
            {"__builtins__": {}},
            values,
        )
    defaults = {
        "conv": EV_PER_AU,
        "fs_to_au": FS_TO_AU,
        "pulse_ampl": 0.001,
        "pulse_freq": 2.0 / EV_PER_AU,
        "pulse_sigm": FS_TO_AU,
        "pulse_varp": 0.0,
        "pulse_t0": 5.0 * FS_TO_AU,
    }
    return {
        name: float(values.get(name, default))
        for name, default in defaults.items()
    }


def process_propagation(
    propagation: Path,
    generator: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return energy and ``-Im[P(omega)/E(omega)]`` on the positive branch."""
    parameters = _pulse_parameters(generator)
    data = np.loadtxt(propagation, dtype=float)
    time = data[:, 0]
    if data.shape[1] < 9:
        raise ValueError(f"Unexpected two-level density-matrix data: {propagation}")

    rho_01 = data[:, 3] + 1.0j * data[:, 4]
    rho_10 = data[:, 5] + 1.0j * data[:, 6]
    dipole = np.real(rho_01 + rho_10)
    tail = max(len(dipole) // 5, 1)
    polarization = dipole - float(np.mean(dipole[-tail:]))

    tau = time - parameters["pulse_t0"]
    sigma = parameters["pulse_sigm"]
    pulse = (
        parameters["pulse_ampl"]
        * np.pi
        / (np.sqrt(2.0 * np.pi) * sigma)
        * np.exp(-0.5 * (tau / sigma) ** 2)
        * np.cos(
            parameters["pulse_freq"] * tau
            + parameters["pulse_varp"] * tau**2
        )
    )

    polarization_frequency = np.fft.fft(polarization)
    pulse_frequency = np.fft.fft(pulse)
    omega = 2.0 * np.pi * np.fft.fftfreq(
        len(time), d=float(time[1] - time[0])
    )
    positive = omega > 0.0
    omega = omega[positive]
    susceptibility = polarization_frequency[positive] / (
        pulse_frequency[positive] + 1.0e-20
    )
    # P/E has inverse-energy units in the atomic-unit propagation.  Divide by
    # eV per Hartree so that its spectral area is unchanged after expressing
    # the horizontal axis in eV.  Do not include the extra omega factor used
    # for absorbed power: the benchmarks compare the susceptibility spectral
    # function with -Im G^R on the same footing.
    susceptibility_spectrum = -np.imag(susceptibility) / parameters["conv"]
    return omega * parameters["conv"], susceptibility_spectrum


def _save_susceptibility_spectrum(
    output: Path,
    energy: np.ndarray,
    spectrum: np.ndarray,
    provenance: str,
) -> None:
    np.savetxt(
        output,
        np.column_stack((energy, spectrum)),
        header=(
            "Energy_eV MinusImChi_eV^-1; -Im[P(omega)/E(omega)] "
            "from np.fft.fft with exp(-i omega t); " + provenance
        ),
    )


def _convert_legacy_power_spectrum(output: Path) -> bool:
    """Convert a retained ``-omega Im chi`` file when propagation is absent."""
    with output.open(encoding="utf-8") as handle:
        header = handle.readline()
    if "MinusImChi_eV^-1" in header:
        return False
    if "Energy_eV Absorption_arb; P/E from np.fft.fft" not in header:
        raise ValueError(f"Unrecognized MF-HEOM spectrum header: {output}")

    data = np.loadtxt(output, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2 or np.any(data[:, 0] <= 0.0):
        raise ValueError(f"Malformed legacy MF-HEOM spectrum: {output}")
    # The retained ordinate is -omega_au Im chi.  Dividing by the energy in
    # eV both removes omega and converts chi from Hartree^-1 to eV^-1.
    spectrum = data[:, 1] / data[:, 0]
    _save_susceptibility_spectrum(
        output,
        data[:, 0],
        spectrum,
        "converted exactly from the retained -omega Im[P/E] array",
    )
    return True


def _case_directories(root: Path) -> Iterable[Path]:
    for child in root.iterdir():
        if not child.is_dir():
            continue
        label = child.name.split("=", 1)[-1]
        try:
            float(label)
        except ValueError:
            continue
        yield child


def process_tree(source_root: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    source_cases = sorted(
        _case_directories(source_root),
        key=lambda path: float(path.name.split("=", 1)[-1]),
    )
    for source in source_cases:
        target_name = source.name
        if source.name.startswith("wc=") and not (target_root / target_name).exists():
            target_name = source.name.split("=", 1)[1]
        if not source.name.startswith("wc=") and (target_root / f"wc={source.name}").exists():
            target_name = f"wc={source.name}"
        target = target_root / target_name
        if not target.exists():
            continue
        propagation = source / "prop-rhom.dat"
        generator = source / "gen_input.py"
        output = target / "absorption_pulse_2eV.txt"
        if not propagation.exists() or not generator.exists():
            if output.exists() and _convert_legacy_power_spectrum(output):
                print(f"Converted {output}")
            continue
        energy, susceptibility_spectrum = process_propagation(
            propagation, generator
        )
        _save_susceptibility_spectrum(
            output,
            energy,
            susceptibility_spectrum,
            "propagated Gaussian carrier and amplitude reconstructed "
            "from gen_input.py",
        )
        print(f"Saved {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    parser.add_argument("target_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_tree(args.source_root.resolve(), args.target_root.resolve())


if __name__ == "__main__":
    main()
