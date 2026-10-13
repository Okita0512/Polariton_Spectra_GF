#!/usr/bin/env python3
"""Load the canonical uncoupled-molecule linewidths used by figures."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


CONFIG_PATH = Path(__file__).with_name("endpoint_linewidths.json")


@dataclass(frozen=True)
class EndpointLinewidths:
    """Canonical direct-FWHM endpoint values, in meV."""

    heom_molecule_fwhm_meV: float
    scba_molecule_fwhm_meV: float
    eq26_gamma_ex_meV: float


def load_endpoint_linewidths(
    path: Path = CONFIG_PATH,
) -> EndpointLinewidths:
    """Read and validate the shared endpoint-linewidth configuration."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    canonical = payload["canonical"]
    values = EndpointLinewidths(
        heom_molecule_fwhm_meV=float(
            canonical["heom_molecule_fwhm_meV"]
        ),
        scba_molecule_fwhm_meV=float(
            canonical["scba_molecule_fwhm_meV"]
        ),
        eq26_gamma_ex_meV=float(canonical["eq26_gamma_ex_meV"]),
    )
    vector = np.asarray(
        (
            values.heom_molecule_fwhm_meV,
            values.scba_molecule_fwhm_meV,
            values.eq26_gamma_ex_meV,
        )
    )
    if not np.all(np.isfinite(vector)) or np.any(vector <= 0.0):
        raise ValueError(f"Invalid endpoint linewidths in {path}")
    if not np.isclose(
        values.eq26_gamma_ex_meV,
        values.heom_molecule_fwhm_meV,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Eq. (26) Gamma_ex must equal the canonical HEOM endpoint"
        )
    return values
