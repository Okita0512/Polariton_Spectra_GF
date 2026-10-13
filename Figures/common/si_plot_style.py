#!/usr/bin/env python3
"""Shared, SI-only plotting conventions for publication vector figures.

This module is intentionally not imported by ``Figures/Main``.  It mirrors
the main-figure Helvetica typography and established color semantics without
changing any frozen main-text plotting behavior.
"""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib as mpl
from matplotlib.axes import Axes


# Method semantics used throughout the SI.
EXACT_COLOR = "#000000"
SCBA_COLOR = "#D62728"
BA_COLOR = "#777777"
HEOM_BLUE = "#5AA9E6"

EXACT_STYLE: Mapping[str, object] = {
    "color": EXACT_COLOR,
    "linestyle": "-",
    "linewidth": 0.90,
    "solid_joinstyle": "round",
    "zorder": 3,
}
SCBA_STYLE: Mapping[str, object] = {
    "color": SCBA_COLOR,
    "linestyle": (0, (4.0, 2.0)),
    "linewidth": 0.90,
    "dash_capstyle": "butt",
    "zorder": 2,
}
BA_STYLE: Mapping[str, object] = {
    "color": BA_COLOR,
    "linestyle": (0, (2.0, 1.6)),
    "linewidth": 0.82,
    "zorder": 1,
}

# Explicitly reproduces the frozen main-Fig.-3 viridis/log(N) mapping.  Keep
# the map explicit so omitted N values never shift the colors in an SI plot.
N_COLORS: Mapping[int, str] = {
    1: "#FDE725",
    2: "#CDE11D",
    4: "#9BD93C",
    8: "#6CCD5A",
    16: "#42BE71",
    32: "#27AD81",
    64: "#1E9C89",
    10000: "#440154",  # collective-limit convention in main Fig. 3
}

# Parameter colors used by the dimer scan/cuts and temperature diagnostics.
S_COLORS: Mapping[float, str] = {
    0.02: "#4C78A8",
    0.05: "#F28E2B",
    0.10: "#4E9F3D",
}
TEMPERATURE_COLORS: Mapping[int, str] = {
    0: "#3B4CC0",
    100: "#6F92D1",
    200: "#E69F00",
    300: "#B40426",
}

PANEL_LABEL_SIZE = 10.4
PANEL_LABEL_OFFSET_PT = (0.0, 4.0)


def configure_si_matplotlib() -> None:
    """Apply compact defaults sized for a nearly full-width SI figure."""

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica",
                "Arial",
                "Liberation Sans",
                "DejaVu Sans",
            ],
            "font.weight": "normal",
            "font.size": 8.4,
            "axes.labelsize": 9.2,
            "axes.labelweight": "normal",
            "axes.titlesize": 9.2,
            "axes.titleweight": "normal",
            "axes.linewidth": 0.65,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 8.3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "pdf.use14corefonts": True,
            "ps.useafm": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "stixsans",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "path.simplify": False,
            "agg.path.chunksize": 0,
        }
    )


def style_axis(
    axis: Axes,
    *,
    top: bool = True,
    right: bool = True,
    minor: bool = True,
) -> None:
    """Apply the shared thin frame and restrained inward ticks to one axis."""

    axis.tick_params(
        which="both",
        direction="in",
        top=top,
        right=right,
        width=0.60,
    )
    axis.tick_params(which="major", length=3.2)
    axis.tick_params(which="minor", length=1.8 if minor else 0.0)


def add_panel_label(
    axis: Axes,
    label: str,
    *,
    x: float = 0.0,
    y: float = 1.0,
    offset_points: tuple[float, float] = PANEL_LABEL_OFFSET_PT,
    size: float = PANEL_LABEL_SIZE,
) -> None:
    """Place a panel letter in the reserved band above an axes rectangle."""

    axis.annotate(
        label,
        xy=(x, y),
        xycoords="axes fraction",
        xytext=offset_points,
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=size,
        fontweight="bold",
        color="black",
        zorder=10,
        annotation_clip=False,
        clip_on=False,
    )


def compact_legend_kwargs() -> dict[str, object]:
    """Return unboxed legend spacing shared by SI plots."""

    return {
        "frameon": False,
        "borderaxespad": 0.25,
        "handlelength": 2.6,
        "handletextpad": 0.55,
        "columnspacing": 1.5,
        "labelspacing": 0.35,
    }
