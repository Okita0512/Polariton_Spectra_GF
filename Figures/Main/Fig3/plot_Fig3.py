#!/usr/bin/env python3
"""Plot the LP-only main-text Fig. 3 from cached detuning data."""

from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
SHARED_DIR = PROJECT_ROOT / "Figures" / "shared" / "detuning"
sys.path.insert(0, str(SHARED_DIR))

from detuning_plot import render_main_lp, write_validation  # noqa: E402


if __name__ == "__main__":
    output = SCRIPT_DIR / "Fig3.pdf"
    validation = render_main_lp(output)
    write_validation(SCRIPT_DIR / "validation.json", validation)
    print(f"Wrote {output.relative_to(PROJECT_ROOT)}")
    print(f"Validation: {validation['validation']}")

