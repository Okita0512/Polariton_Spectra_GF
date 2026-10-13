#!/usr/bin/env python3
"""Plot-only rebuild of main Fig. 2 from the existing cached results."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "main.tex").is_file():
            return candidate
    raise RuntimeError("Could not locate project root containing main.tex")


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
sys.path.insert(0, str(PROJECT_ROOT / "Figures" / "common"))
from area_normalization import match_area_to_reference  # noqa: E402


OUTPUT = SCRIPT_DIR / "Fig2.pdf"
VALIDATION = SCRIPT_DIR / "figure_validation.txt"
EV_PER_AU = 27.211397
PANEL_LABEL_POSITION = (-0.15, 0.985)
PANEL_LABEL_SIZE = 10.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pair(relative: str, molecule_count: int):
    directory = SCRIPT_DIR / relative
    heom_energy = np.loadtxt(directory / "resp1st.w1") * EV_PER_AU
    heom = np.loadtxt(directory / "resp1st_im.w") / (molecule_count * EV_PER_AU)
    scba_data = np.loadtxt(directory / "Aw_sc_fullpol.txt")
    scba_energy = scba_data[:, 0] * EV_PER_AU
    scba = match_area_to_reference(
        heom_energy, heom, scba_energy, -scba_data[:, 1] / EV_PER_AU
    )
    return heom_energy, heom, scba_energy, scba


def style_axis(axis):
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.tick_params(which="major", length=3.2, width=0.7)
    axis.tick_params(which="minor", length=1.8, width=0.55)
    axis.xaxis.set_minor_locator(AutoMinorLocator(2))
    axis.yaxis.set_minor_locator(AutoMinorLocator(2))


def add_panel_label(axis, panel):
    axis.text(*PANEL_LABEL_POSITION, f"({panel})", transform=axis.transAxes,
              ha="right", va="top", fontsize=PANEL_LABEL_SIZE,
              clip_on=False)


def plot_spectra(axis, cases, ylim, panel, bath_name):
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(cases)))
    ratios = []
    for color, (relative, count, label) in zip(colors, cases):
        x_ref, y_ref, x_scba, y_scba = load_pair(relative, count)
        axis.fill_between(x_ref, y_ref, 0.0, color="lightskyblue", alpha=0.28, lw=0)
        axis.plot(x_scba, y_scba, color=color, lw=0.8, label=label)
        ref_area = np.trapz(y_ref, x_ref)
        scba_area = np.trapz(y_scba, x_scba)
        ratios.append(scba_area / ref_area)
    axis.set_xlim(1.75, 2.30)
    axis.set_ylim(*ylim)
    axis.xaxis.set_major_locator(MultipleLocator(0.1))
    axis.set_xlabel(r"$\omega$ (eV)")
    axis.set_ylabel(r"$A(\omega)/N$ (eV$^{-1}$)")
    add_panel_label(axis, panel)
    axis.text(0.97, 0.95, bath_name, transform=axis.transAxes,
              ha="right", va="top", fontsize=7.2)
    style_axis(axis)
    return ratios


def load_widths(bath: str, method: str):
    path = SCRIPT_DIR / f"N-scaling_{bath}" / f"width-{method}.txt"
    data = np.loadtxt(path)
    n_values = data[:, 0]
    lp_mev = 1000.0 * data[:, 1]
    valid = np.isfinite(n_values) & np.isfinite(lp_mev) & (n_values > 0) & (lp_mev > 0)
    return path, np.log10(n_values[valid]), np.log10(lp_mev[valid])


def fit_line(x, y, selection):
    slope, intercept = np.polyfit(x[selection], y[selection], 1)
    x_fit = np.linspace(float(x[selection].min()), float(x[selection].max()), 80)
    return slope, x_fit, slope * x_fit + intercept


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.2,
    "axes.labelsize": 8.2,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
    "legend.fontsize": 5.9,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
})

fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.25), constrained_layout=True)
fig.get_layout_engine().set(w_pad=0.025, h_pad=0.02, wspace=0.035)

fast_cases = (
    ("fast/N=1_out", 1, "Bare molecule"),
    ("fast/N=1", 1, r"$N=1$"),
    ("fast/N=2", 2, r"$N=2$"),
    ("fast/N=4", 4, r"$N=4$"),
    ("fast/N=8", 8, r"$N=8$"),
)
slow_cases = (
    ("slow/outside", 1, "Bare molecule"),
    ("slow/N=1", 1, r"$N=1$"),
    ("slow/N=2", 2, r"$N=2$"),
    ("slow/N=4", 4, r"$N=4$"),
    ("slow/N=8", 8, r"$N=8$"),
)
fast_ratios = plot_spectra(axes[0], fast_cases, (-10, 270), "a", "fast bath")
slow_ratios = plot_spectra(axes[1], slow_cases, (-5, 100), "b", "slow bath")
axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.55, 0.88),
               ncol=2, columnspacing=0.75, handlelength=1.5, handletextpad=0.35)

slow_heom_path, x_sh, y_sh = load_widths("slow", "HEOM")
slow_scba_path, x_ss, y_ss = load_widths("slow", "SCBA")
fast_heom_path, x_fh, y_fh = load_widths("fast", "HEOM")
fast_scba_path, x_fs, y_fs = load_widths("fast", "SCBA")

axis = axes[2]
axis.plot(x_sh, y_sh, "o", ms=3.0, mfc="white", mec="tab:blue", mew=0.8,
          label="slow, HEOM")
axis.plot(x_ss, y_ss, "o-", ms=2.3, lw=0.8, color="tab:blue", label="slow, SCBA")
axis.plot(x_fh, y_fh, "s", ms=2.8, mfc="white", mec="tab:red", mew=0.8,
          label="fast, HEOM")
axis.plot(x_fs, y_fs, "s-", ms=2.2, lw=0.8, color="tab:red", label="fast, SCBA")

slow_slope, slow_xfit, slow_yfit = fit_line(x_ss, y_ss, slice(0, 4))
fast_slope, fast_xfit, fast_yfit = fit_line(x_fs, y_fs, slice(7, 12))
if not np.isclose(slow_slope, -0.543196, atol=5e-6):
    raise RuntimeError(f"Slow-bath fit changed: {slow_slope}")
if not np.isclose(fast_slope, -0.987177, atol=5e-6):
    raise RuntimeError(f"Fast-bath fit changed: {fast_slope}")
axis.plot(slow_xfit, slow_yfit, "--", lw=0.8, color="navy", label=r"slow fit, $k=-0.543$")
axis.plot(fast_xfit, fast_yfit, "--", lw=0.8, color="darkred", label=r"fast fit, $k=-0.987$")
axis.set_xlabel(r"$\log_{10}N$")
axis.set_ylabel(r"$\log_{10}(\Gamma_{\rm LP}/{\rm meV})$")
axis.xaxis.set_major_locator(MultipleLocator(0.5))
axis.yaxis.set_major_locator(MultipleLocator(0.5))
axis.set_xlim(min(x_ss.min(), x_fs.min()) - 0.08, max(x_ss.max(), x_fs.max()) + 0.08)
axis.set_ylim(min(y_ss.min(), y_fs.min()) - 0.16, max(y_ss.max(), y_fs.max()) + 0.22)
add_panel_label(axis, "c")
axis.legend(frameon=False, loc="lower left", ncol=1, handlelength=1.5,
            labelspacing=0.22, handletextpad=0.35)
style_axis(axis)

fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.015)
plt.close(fig)

sources = [
    SCRIPT_DIR / relative / name
    for relative, _, _ in (*fast_cases, *slow_cases)
    for name in ("resp1st.w1", "resp1st_im.w", "Aw_sc_fullpol.txt")
] + [slow_heom_path, slow_scba_path, fast_heom_path, fast_scba_path]
VALIDATION.write_text(
    "\n".join([
        "Main Fig. 2 plot-only validation",
        "status = PASS",
        "panels = fast-bath spectra; slow-bath spectra; fixed-collective-splitting LP scaling",
        f"slow fit (N=1--4) = {slow_slope:.9f}",
        f"fast fit (N=8--128) = {fast_slope:.9f}",
        f"max area-ratio error, spectral pairs = {max(abs(np.asarray(fast_ratios + slow_ratios) - 1.0)):.3e}",
        "sources:",
        *[f"{path.relative_to(PROJECT_ROOT)} sha256={sha256(path)}" for path in sources],
    ]) + "\n",
    encoding="utf-8",
)
print(f"Wrote {OUTPUT.relative_to(PROJECT_ROOT)}")
print(f"Wrote {VALIDATION.relative_to(PROJECT_ROOT)}")
