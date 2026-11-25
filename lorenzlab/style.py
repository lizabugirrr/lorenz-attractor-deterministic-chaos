"""Єдиний стиль графіків, узгоджений з темою презентації."""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

NAVY = "#14213D"
ORANGE = "#E76F51"
GOLD = "#F4A261"
TEAL = "#2A9D8F"
SAND = "#E9C46A"
GREY = "#8D99AE"
PURPLE = "#6D597A"

BUTTERFLY = LinearSegmentedColormap.from_list(
    "butterfly", ["#14213D", "#2A6F97", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51"])

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")


def setup():
    for f in glob.glob(os.path.expanduser(
            "~/.TinyTeX/texmf-dist/fonts/opentype/public/fira/FiraSans-*.otf")):
        try:
            fm.fontManager.addfont(f)
        except Exception:
            pass
    fam = "Fira Sans" if any("Fira Sans" == f.name for f in fm.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({
        "font.family": fam,
        "mathtext.fontset": "dejavusans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=[NAVY, ORANGE, TEAL, GOLD, PURPLE, GREY]),
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "pdf.fonttype": 42,
    })
    os.makedirs(FIG_DIR, exist_ok=True)


def save(fig, name, png=False):
    path = os.path.join(FIG_DIR, name + (".png" if png else ".pdf"))
    fig.savefig(path)
    plt.close(fig)
    print("  ->", os.path.relpath(path))
    return path
