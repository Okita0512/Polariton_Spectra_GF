#!/usr/bin/env python3
"""Regenerate data, audits, validation, and figures."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> None:
    root = Path(__file__).resolve().parent
    for script in (
        "calculate_exact_cumulant.py",
        "calculate_scba.py",
        "calculate_single_shot_ba.py",
        "check_convergence.py",
        "validate_outputs.py",
        "plot_FigS3.py",
    ):
        print(f"\nRunning {script}", flush=True)
        subprocess.run(
            [sys.executable, str(root / script)],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    main()
