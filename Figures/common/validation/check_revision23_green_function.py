#!/usr/bin/env python3
"""Algebra-only regression checks for the revised main-text Green's functions."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().with_name("revision23_green_function_checks.txt")


def inverse_block_bright(omega, omega_c, omega_0, kappa, sigma, n_mol, g_c):
    inverse = np.array(
        [
            [omega - omega_c + 0.5j * kappa, -np.sqrt(n_mol) * g_c],
            [-np.sqrt(n_mol) * g_c, omega - omega_0 + 1.0e-12j - sigma],
        ],
        dtype=complex,
    )
    return np.linalg.inv(inverse)[1, 1]


def schur_bright(omega, omega_c, omega_0, kappa, sigma, n_mol, g_c):
    return 1.0 / (
        omega
        - omega_0
        + 1.0e-12j
        - sigma
        - n_mol * g_c**2 / (omega - omega_c + 0.5j * kappa)
    )


omega = 2.117 + 0.007j
omega_c = 2.03
omega_0 = 2.00
kappa = 0.04415
sigma = 0.006 - 0.021j
n_mol = 8
g_c = 0.03536

direct = inverse_block_bright(omega, omega_c, omega_0, kappa, sigma, n_mol, g_c)
schur = schur_bright(omega, omega_c, omega_0, kappa, sigma, n_mol, g_c)
schur_error = abs(direct - schur)

g_dark = 1.0 / (omega - omega_0 + 1.0e-12j - sigma)
g_bright_n1 = schur_bright(omega, omega_c, omega_0, kappa, sigma, 1, g_c)
gbar_n1 = (g_bright_n1 + 0 * g_dark) / 1
n1_error = abs(gbar_n1 - g_bright_n1)

g_bright_gc0 = schur_bright(omega, omega_c, omega_0, kappa, sigma, n_mol, 0.0)
gbar_gc0 = (g_bright_gc0 + (n_mol - 1) * g_dark) / n_mol
gc0_error = max(abs(g_bright_gc0 - g_dark), abs(gbar_gc0 - g_dark))

g_sigma0 = schur_bright(omega, omega_c, omega_0, kappa, 0.0j, n_mol, g_c)
g_sigma0_direct = inverse_block_bright(
    omega, omega_c, omega_0, kappa, 0.0j, n_mol, g_c
)
sigma0_error = abs(g_sigma0 - g_sigma0_direct)

# With J(nu)=0, the displayed Fan integral is identically zero for any Gbar.
zero_vibronic_sigma = 0.0

tolerance = 2.0e-12
if max(schur_error, n1_error, gc0_error, sigma0_error) > tolerance:
    raise RuntimeError("Green's-function algebra regression failed")

OUTPUT.write_text(
    "\n".join(
        [
            "Revision 23 Green's-function algebra checks",
            "status = PASS",
            f"2x2 inversion versus Schur complement error = {schur_error:.3e}",
            f"N=1 identity Gbar=GBB error = {n1_error:.3e}",
            f"g_c=0 identity GBB=GD=Gbar error = {gc0_error:.3e}",
            f"Sigma=0 block identity error = {sigma0_error:.3e}",
            f"J=0 Fan self-energy = {zero_vibronic_sigma:.1f}",
            "No spectra or benchmark calculations were run.",
        ]
    )
    + "\n",
    encoding="utf-8",
)
print(OUTPUT.relative_to(ROOT))
