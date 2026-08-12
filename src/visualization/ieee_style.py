# -*- coding: utf-8 -*-
"""
IEEE IEEEtran figure style for matplotlib.

Applies publication-ready defaults conformant with IEEE Transactions
(IEEEtran) typography and figure guidelines:

  - DPI             : 600 (IEEE minimum for raster figures)
  - Font            : Times New Roman 10pt (IEEEtran body default)
  - Math            : STIX (Times-compatible math font)
  - Title           : 11pt bold
  - Axis labels     : 10pt bold
  - Figure size     : 7.16 x 4.5 in (IEEE double-column width)
  - Grid            : dashed, alpha=0.25, linewidth=0.5
  - Ticks           : inward
  - Legend          : framed, alpha=0.9
  - Spines          : partial (top/right hidden)
  - Padding         : 0.05 layout pad

Usage
-----
    from src.visualization.ieee_style import apply_ieee_style
    apply_ieee_style()   # call once at import-time in a figure module

Per-axes polish (ticks inward, hide top/right spines) can be applied
via `ieee_axes(ax)` after plotting.
"""
from __future__ import annotations

import matplotlib
# Non-interactive backend for headless runs (safe on Windows too)
try:
    matplotlib.use("Agg")
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------------------------------------------------------------
# IEEE IEEEtran constants
# ---------------------------------------------------------------------------
IEEE_DPI = 600                        # minimum for raster figures
IEEE_FONT_FAMILY = "serif"
IEEE_FONT_SERIF = [
    "Times New Roman", "Times", "STIX", "DejaVu Serif"
]
IEEE_FONT_SIZE = 10                   # body text
IEEE_TITLE_SIZE = 11                  # title
IEEE_LABEL_SIZE = 10                  # axis labels
IEEE_TICK_SIZE = 9                    # tick labels
IEEE_LEGEND_SIZE = 9                  # legend text

IEEE_DOUBLE_COL = (7.16, 4.5)         # double-column figure (inches)
IEEE_SINGLE_COL = (3.5, 2.6)          # single-column figure
IEEE_SQUARE = (3.5, 3.5)              # square single-column

IEEE_GRID_ALPHA = 0.25
IEEE_GRID_LW = 0.5
IEEE_LEGEND_ALPHA = 0.9
IEEE_LAYOUT_PAD = 0.05

_APPLIED = False


def apply_ieee_style() -> None:
    """Install IEEE IEEEtran defaults into matplotlib rcParams.

    Idempotent: safe to call multiple times. Subsequent calls are no-ops.
    """
    global _APPLIED
    if _APPLIED:
        return

    # --- Font / text ---
    rcParams["font.family"] = IEEE_FONT_FAMILY
    rcParams["font.serif"] = IEEE_FONT_SERIF
    rcParams["font.size"] = IEEE_FONT_SIZE
    rcParams["text.usetex"] = False                # keep headless portability
    rcParams["mathtext.fontset"] = "stix"          # Times-compatible math
    rcParams["mathtext.default"] = "regular"

    # --- Titles / labels ---
    rcParams["axes.titlesize"] = IEEE_TITLE_SIZE
    rcParams["axes.titleweight"] = "bold"
    rcParams["axes.labelsize"] = IEEE_LABEL_SIZE
    rcParams["axes.labelweight"] = "bold"
    rcParams["xtick.labelsize"] = IEEE_TICK_SIZE
    rcParams["ytick.labelsize"] = IEEE_TICK_SIZE

    # --- Figure ---
    rcParams["figure.figsize"] = IEEE_DOUBLE_COL
    rcParams["figure.dpi"] = 100                   # on-screen; savefig uses 600
    rcParams["figure.autolayout"] = False          # we control with tight_layout
    rcParams["figure.constrained_layout.use"] = False

    # --- Saving ---
    rcParams["savefig.dpi"] = IEEE_DPI
    rcParams["savefig.bbox"] = "tight"
    rcParams["savefig.pad_inches"] = IEEE_LAYOUT_PAD
    rcParams["savefig.format"] = "png"

    # --- Axes / spines ---
    rcParams["axes.linewidth"] = 0.8
    rcParams["axes.edgecolor"] = "#333333"
    rcParams["axes.spines.top"] = False            # partial spines
    rcParams["axes.spines.right"] = False
    rcParams["axes.spines.left"] = True
    rcParams["axes.spines.bottom"] = True
    rcParams["axes.axisbelow"] = True              # grid below data

    # --- Grid ---
    rcParams["axes.grid"] = True
    rcParams["grid.linestyle"] = "--"
    rcParams["grid.linewidth"] = IEEE_GRID_LW
    rcParams["grid.alpha"] = IEEE_GRID_ALPHA
    rcParams["grid.color"] = "#888888"

    # --- Ticks (inward) ---
    rcParams["xtick.direction"] = "in"
    rcParams["ytick.direction"] = "in"
    rcParams["xtick.major.size"] = 3.5
    rcParams["ytick.major.size"] = 3.5
    rcParams["xtick.minor.size"] = 2.0
    rcParams["ytick.minor.size"] = 2.0
    rcParams["xtick.major.width"] = 0.7
    rcParams["ytick.major.width"] = 0.7
    rcParams["xtick.top"] = False
    rcParams["ytick.right"] = False

    # --- Legend (framed, alpha=0.9) ---
    rcParams["legend.fontsize"] = IEEE_LEGEND_SIZE
    rcParams["legend.frameon"] = True
    rcParams["legend.framealpha"] = IEEE_LEGEND_ALPHA
    rcParams["legend.edgecolor"] = "#333333"
    rcParams["legend.fancybox"] = False
    rcParams["legend.borderpad"] = 0.4
    rcParams["legend.labelspacing"] = 0.3
    rcParams["legend.handlelength"] = 1.6
    rcParams["legend.handletextpad"] = 0.5
    rcParams["legend.columnspacing"] = 1.0

    # --- Lines / markers ---
    rcParams["lines.linewidth"] = 1.2
    rcParams["lines.markersize"] = 4.0
    rcParams["patch.linewidth"] = 0.5
    rcParams["hatch.linewidth"] = 0.5

    # --- Error bars / caps ---
    rcParams["errorbar.capsize"] = 2.0

    # --- PDF/PS embedding (for IEEE submission tooling) ---
    rcParams["pdf.fonttype"] = 42                  # TrueType
    rcParams["ps.fonttype"] = 42

    _APPLIED = True


def ieee_axes(ax, *, grid: bool = True,
              hide_top: bool = True, hide_right: bool = True) -> None:
    """Apply per-axes IEEE polish: inward ticks, partial spines, dashed grid.

    Useful for axes created before ``apply_ieee_style`` was called, or
    for 3D / polar axes where some rcParams don't propagate.
    """
    ax.tick_params(direction="in", which="both",
                   length=3.5, width=0.7)
    if hide_top and "top" in ax.spines:
        ax.spines["top"].set_visible(False)
    if hide_right and "right" in ax.spines:
        ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, linestyle="--", linewidth=IEEE_GRID_LW,
                alpha=IEEE_GRID_ALPHA, color="#888888")
        ax.set_axisbelow(True)


def ieee_legend(ax, *args, **kwargs):
    """Create a framed IEEE legend (alpha=0.9, thin edge)."""
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("framealpha", IEEE_LEGEND_ALPHA)
    kwargs.setdefault("edgecolor", "#333333")
    kwargs.setdefault("fancybox", False)
    kwargs.setdefault("fontsize", IEEE_LEGEND_SIZE)
    leg = ax.legend(*args, **kwargs)
    if leg is not None:
        frame = leg.get_frame()
        frame.set_linewidth(0.6)
    return leg


def ieee_savefig(fig, path, *, dpi: int = IEEE_DPI,
                 pad: float = IEEE_LAYOUT_PAD) -> None:
    """Save a figure with IEEE defaults (600 dpi, tight bbox, 0.05 pad)."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad)


# Auto-apply on import — modules that `from .ieee_style import ...` get
# IEEE defaults for free. Explicit `apply_ieee_style()` calls remain idempotent.
apply_ieee_style()
