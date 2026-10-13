#!/usr/bin/env python3
"""Compatibility wrapper for the shared linewidth analysis."""
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (
            (candidate / "main.tex").is_file()
            and (candidate / "Figures" / "common" / "rebuild_analysis.py").is_file()
        ):
            return candidate
    raise RuntimeError(f"Could not locate the project root above {start}")


ROOT = find_project_root(SCRIPT_DIR)


def main() -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "Figures" / "common" / "rebuild_analysis.py")],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
